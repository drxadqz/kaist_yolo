# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
#from mmcv.ops import DeformConv2dPack as DCN

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        #self.m = nn.ModuleList(DCN(x, self.no * self.na, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deform_groups=1) for x in ch)  # output DCN conv3x3

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        logits_ = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                logits = x[i][..., 5:]

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
                logits_.append(logits.view(bs, -1, self.no - 5))

        return x if self.training else (torch.cat(z, 1), torch.cat(logits_, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class QualityDetect(Detect):
    """Detect head with an auxiliary quality branch for low-FPPI ranking.

    The extra branch predicts a countability-aware quality score per anchor cell.
    During inference, this score modulates objectness so high-risk ambiguous boxes
    are ranked lower without changing the box regression/classification pipeline.
    """

    def __init__(self, nc=80, anchors=(), quality_gain=0.75, ch=()):
        super(QualityDetect, self).__init__(nc=nc, anchors=anchors, ch=ch)
        self.quality_gain = float(quality_gain)
        self.quality_m = nn.ModuleList(nn.Conv2d(x, self.na, 1) for x in ch)
        self.use_quality = True
        self.last_quality = None

    def __getstate__(self):
        # Runtime caches are useful for ranking loss/debugging but must not
        # participate in checkpoint deepcopy/save.
        state = super().__getstate__()
        state['last_quality'] = None
        return state

    def forward(self, x):
        z = []
        logits_ = []
        self.training |= self.export
        self.last_quality = []
        for i in range(self.nl):
            quality = self.quality_m[i](x[i])
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            quality = quality.view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            self.last_quality.append(quality)

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                logits = x[i][..., 5:]

                y = x[i].sigmoid()
                quality_prob = quality.sigmoid()
                quality_factor = (1.0 - self.quality_gain) + self.quality_gain * quality_prob
                y[..., 4:5] *= quality_factor
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
                logits_.append(logits.view(bs, -1, self.no - 5))

        return x if self.training else (torch.cat(z, 1), torch.cat(logits_, 1), x)


class ResidualQualityDetect(Detect):
    """Residual-neutral countability head for low-FPPI score ranking.

    The extra branch predicts q with shape [B, na, H, W, 1] for each detection
    scale. It never changes bbox/obj/cls logits. Inference calibration is only
    active when `apply_in_inference` is true and alpha > 0:

        factor = clamp(1 + alpha * tanh(q), factor_min, factor_max)

    With alpha=0 or apply_in_inference=False, the module is behaviorally
    equivalent to Detect while still exposing q for loss/dry-run checks.
    """

    def __init__(self, nc=80, anchors=(), alpha=0.0, factor_min=0.95, factor_max=1.05,
                 apply_in_inference=False, ch=()):
        super(ResidualQualityDetect, self).__init__(nc=nc, anchors=anchors, ch=ch)
        self.countability_alpha = float(alpha)
        self.countability_factor_min = float(factor_min)
        self.countability_factor_max = float(factor_max)
        self.countability_apply_in_inference = bool(apply_in_inference)
        self.quality_m = nn.ModuleList(nn.Conv2d(x, self.na, 1) for x in ch)
        self.use_quality = True
        self.use_countability_calibration = True
        self.last_quality = None
        self.last_quality_factor = None
        self._init_quality_branch()

    def __getstate__(self):
        # Runtime q caches are forward products, not persistent model state.
        # Removing them from deepcopy prevents checkpoint save failures.
        state = super().__getstate__()
        state['last_quality'] = None
        state['last_quality_factor'] = None
        return state

    def _init_quality_branch(self):
        # Neutral init: q=0, so factor=1 whenever residual calibration is enabled.
        for conv in self.quality_m:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def set_countability_runtime_config(self, alpha=None, factor_min=None, factor_max=None, apply_in_inference=None):
        if alpha is not None:
            self.countability_alpha = float(alpha)
        if factor_min is not None:
            self.countability_factor_min = float(factor_min)
        if factor_max is not None:
            self.countability_factor_max = float(factor_max)
        if apply_in_inference is not None:
            self.countability_apply_in_inference = bool(apply_in_inference)

    def forward(self, x):
        z = []
        logits_ = []
        self.training |= self.export
        self.last_quality = []
        self.last_quality_factor = []
        for i in range(self.nl):
            quality = self.quality_m[i](x[i])
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # q logits are anchor-aligned: [B, na, H, W, 1].
            quality = quality.view(bs, self.na, 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            self.last_quality.append(quality)

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                logits = x[i][..., 5:]
                y = x[i].sigmoid()
                if self.countability_apply_in_inference and self.countability_alpha > 0.0:
                    factor = 1.0 + self.countability_alpha * quality.tanh()
                    factor = factor.clamp(self.countability_factor_min, self.countability_factor_max)
                    y[..., 4:5] *= factor
                else:
                    factor = torch.ones_like(quality)
                self.last_quality_factor.append(factor)
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
                logits_.append(logits.view(bs, -1, self.no - 5))

        return x if self.training else (torch.cat(z, 1), torch.cat(logits_, 1), x)


class FactorizedProtocolDetect(Detect):
    """Explicit H/C/G/U factorized detection head for protocol-semantic ranking.

    The head predicts anchor-aligned logits for:

    - H: humanness
    - C: base countability
    - G: groupness
    - U: uncertainty
    - B: protocol bias

    and forms

        q = c + tanh(b) - alpha_G * sigmoid(g) - alpha_U * sigmoid(u)
        C = sigmoid(q)
        S_sem = sigmoid(h) * C

    The detector bbox/cls/objectness path stays unchanged during training.
    During inference, `S_sem` only applies a bounded residual score factor.
    """

    def __init__(self, nc=80, anchors=(), alpha=0.0, factor_min=0.96, factor_max=1.05,
                 apply_in_inference=False, semantic_center=0.25, score_beta=4.0,
                 alpha_g_init=0.70, alpha_u_init=0.15, ch=()):
        super(FactorizedProtocolDetect, self).__init__(nc=nc, anchors=anchors, ch=ch)
        self.countability_alpha = float(alpha)
        self.countability_factor_min = float(factor_min)
        self.countability_factor_max = float(factor_max)
        self.countability_apply_in_inference = bool(apply_in_inference)
        self.semantic_center = float(semantic_center)
        self.semantic_score_beta = float(score_beta)
        self.factor_m = nn.ModuleList(nn.Conv2d(x, self.na * 5, 1) for x in ch)
        self.alpha_g_raw = nn.Parameter(torch.tensor(self._inv_softplus(alpha_g_init), dtype=torch.float32))
        self.alpha_u_raw = nn.Parameter(torch.tensor(self._inv_softplus(alpha_u_init), dtype=torch.float32))
        self.use_protocol_factorized = True
        self.use_countability_calibration = True
        self.last_factorized = None
        self.last_quality = None
        self.last_quality_factor = None
        self._init_factorized_branch()

    def __getstate__(self):
        state = super().__getstate__()
        state['last_factorized'] = None
        state['last_quality'] = None
        state['last_quality_factor'] = None
        return state

    @staticmethod
    def _inv_softplus(value):
        value = max(float(value), 1e-4)
        return math.log(math.expm1(value))

    def _init_factorized_branch(self):
        for conv in self.factor_m:
            nn.init.zeros_(conv.weight)
            bias = conv.bias.view(self.na, 5)
            nn.init.zeros_(bias[:, 0])
            nn.init.zeros_(bias[:, 1])
            nn.init.constant_(bias[:, 2], -4.0)
            nn.init.constant_(bias[:, 3], -4.0)
            nn.init.zeros_(bias[:, 4])

    def set_countability_runtime_config(self, alpha=None, factor_min=None, factor_max=None, apply_in_inference=None):
        self.set_protocol_factorized_runtime_config(
            alpha=alpha,
            factor_min=factor_min,
            factor_max=factor_max,
            apply_in_inference=apply_in_inference,
        )

    def set_protocol_factorized_runtime_config(
        self,
        alpha=None,
        factor_min=None,
        factor_max=None,
        apply_in_inference=None,
        semantic_center=None,
        score_beta=None,
    ):
        if alpha is not None:
            self.countability_alpha = float(alpha)
        if factor_min is not None:
            self.countability_factor_min = float(factor_min)
        if factor_max is not None:
            self.countability_factor_max = float(factor_max)
        if apply_in_inference is not None:
            self.countability_apply_in_inference = bool(apply_in_inference)
        if semantic_center is not None:
            self.semantic_center = float(semantic_center)
        if score_beta is not None:
            self.semantic_score_beta = float(score_beta)

    def forward(self, x):
        z = []
        logits_ = []
        self.training |= self.export
        self.last_factorized = []
        self.last_quality = []
        self.last_quality_factor = []
        alpha_g = F.softplus(self.alpha_g_raw)
        alpha_u = F.softplus(self.alpha_u_raw)

        for i in range(self.nl):
            factorized = self.factor_m[i](x[i])
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            factorized = factorized.view(bs, self.na, 5, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            h_logit = factorized[..., 0:1]
            c_logit = factorized[..., 1:2]
            g_logit = factorized[..., 2:3]
            u_logit = factorized[..., 3:4]
            b_logit = factorized[..., 4:5]

            h_prob = h_logit.sigmoid()
            g_prob = g_logit.sigmoid()
            u_prob = u_logit.sigmoid()
            b_term = b_logit.tanh()
            q_logit = c_logit + b_term - alpha_g * g_prob - alpha_u * u_prob
            c_prob = q_logit.sigmoid()
            semantic_prob = h_prob * c_prob

            if self.countability_apply_in_inference and self.countability_alpha > 0.0:
                factor = 1.0 + self.countability_alpha * torch.tanh(
                    self.semantic_score_beta * (semantic_prob - self.semantic_center)
                )
                factor = factor.clamp(self.countability_factor_min, self.countability_factor_max)
            else:
                factor = torch.ones_like(semantic_prob)

            self.last_factorized.append({
                'h_logit': h_logit,
                'c_logit': c_logit,
                'g_logit': g_logit,
                'u_logit': u_logit,
                'b_logit': b_logit,
                'bias_term': b_term,
                'q_logit': q_logit,
                'humanness': h_prob,
                'countability': c_prob,
                'groupness': g_prob,
                'uncertainty': u_prob,
                'semantic': semantic_prob,
                'factor': factor,
            })
            self.last_quality.append(q_logit)
            self.last_quality_factor.append(factor)

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                logits = x[i][..., 5:]
                y = x[i].sigmoid()
                y[..., 4:5] *= factor
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
                logits_.append(logits.view(bs, -1, self.no - 5))

        return x if self.training else (torch.cat(z, 1), torch.cat(logits_, 1), x)


class DecoupledDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(DecoupledDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.stems = nn.ModuleList([Conv(x, 256, 1, 1) for x in ch])
        self.cls_convs = nn.ModuleList([nn.Sequential(Conv(256, 256, 3, 1),
                                                      Conv(256, 256, 3, 1)) for _ in ch])
        self.reg_convs = nn.ModuleList([nn.Sequential(Conv(256, 256, 3, 1),
                                                      Conv(256, 256, 3, 1)) for _ in ch])
        self.cls_preds = nn.ModuleList([nn.Conv2d(256, self.nc*self.na, 1, 1) for _ in ch])
        self.reg_preds = nn.ModuleList([nn.Conv2d(256, 4*self.na, 1, 1) for _ in ch])
        self.obj_preds = nn.ModuleList([nn.Conv2d(256, 1*self.na, 1, 1) for _ in ch])


    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        logits_ = []
        self.training |= self.export
        for i in range(self.nl):
            stem_output = self.stems[i](x[i])  # stem conv
            bs, _, ny, nx = x[i].shape

            cls_feat = self.cls_convs[i](stem_output) # cls branch
            cls_output = self.cls_preds[i](cls_feat).view(bs, self.na, self.nc, ny, nx) # [bs, nc*na, ny, nx] -> [bs, na, nc, ny, nx]

            reg_feat = self.reg_convs[i](stem_output) # reg branch
            reg_output = self.reg_preds[i](reg_feat).view(bs, self.na, 4, ny, nx) # [bs, 4*na, ny, nx] -> [bs, na, 4, ny, nx]
            obj_output = self.obj_preds[i](reg_feat).view(bs, self.na, 1, ny, nx) # [bs, na, ny, nx] -> [bs, na, 1, ny, nx]

            x[i] = torch.cat([reg_output, obj_output, cls_output], dim=2) # [bs, na, nc+5, ny, nx]
            x[i] = x[i].permute(0, 1, 3, 4, 2).contiguous() # [bs, na, ny, nx, nc+5

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                logits = x[i][..., 5:]

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
                logits_.append(logits.view(bs, -1, self.no - 5))

        return x if self.training else (torch.cat(z, 1), torch.cat(logits_, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict

        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        # print(m)

        if isinstance(m, Detect) or isinstance(m, DecoupledDetect):
            s = 256  # 2x min stride
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.Tensor([8.0, 16.0, 32.0])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            #self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, x2, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, x2, profile)  # single-scale inference, train


    def forward_once(self, x, x2, profile=False):
        y, dt = [], []  # outputs
        i = 0
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                if m.f != -4:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            if m.f == -4:
                x = m(x2)
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            i += 1

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:

            if m is Focus:
                c1, c2 = 3, args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            elif m is Conv and args[0] == 64:    # new
                c1, c2 = 3, args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            else:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR]:
                    args.insert(2, n)  # number of repeats
                    n = 1

        elif m is ResNetlayer:
            if args[3] == True:
                c2 = args[1]
            else:
                c2 = args[1]*4
        elif m is VGGblock:
            c2 = args[2]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [Add, DMAF]:
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add2:
            c2 = ch[f[0]]
            args = [c2, args[1]]
        elif (m is Detect) or (m is DecoupledDetect) or (m is QualityDetect) or (m is ResidualQualityDetect) or (m is FactorizedProtocolDetect):
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is NiNfusion:
            c1 = sum([ch[x] for x in f])
            c2 = c1 // 2
            args = [c1, c2, *args]
        elif m in [DeformScaledDotTransformerFusionBlockLocal, DeformConsensusDetailFusionBlockLocal,
                   DeformDASRFusionBlockLocal, DeformAIMSFusionBlockLocal, DeformTACMFusionBlockLocal]:
            c2 = ch[f[0]]
            args = [c2, *args[1:]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []

        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/models/ia_dasr_formal.yaml',
                        help='model YAML path')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)
    print(device)


    model = Model(opt.cfg).to(device)
    input_rgb = torch.zeros(1, 3, 128, 128, device=device)
    input_ir = torch.zeros(1, 3, 128, 128, device=device)

    output = model(input_rgb, input_ir)
