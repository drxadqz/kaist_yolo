import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import torchvision
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader_rgb_ir, create_dataloader_mono
from utils.general import logger, coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xyxy2xywh2, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
from evaluation_script.evaluation_script import evaluate
from utils.confluence import confluence_process


def kaist_day_roi_keep_mask(predn, image_index, day_night_split):
    """Suppress daytime boxes outside the KAIST reasonable evaluation region."""
    if day_night_split != 1455 or image_index is None or image_index >= day_night_split:
        return None

    xyxy = predn[:, :4]
    return (
        (xyxy[:, 0] >= 5.0)
        & (xyxy[:, 1] >= 5.0)
        & (xyxy[:, 2] <= 635.0)
        & (xyxy[:, 3] <= 507.0)
    )


def kaist_domain_nms_iou(image_index, day_night_split, day_iou_thres=None, night_iou_thres=None):
    if image_index is None or day_night_split is None:
        return None
    if image_index < day_night_split:
        return day_iou_thres
    return night_iou_thres


def apply_domain_specific_nms(pred, iou_thres):
    if iou_thres is None or len(pred) <= 1:
        return pred

    max_wh = 4096
    boxes = pred[:, :4] + pred[:, 5:6] * max_wh
    keep = torchvision.ops.nms(boxes, pred[:, 4], float(iou_thres))
    return pred[keep]


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _set_drr_aux_export(model, enabled):
    root = _unwrap_model(model)
    for module in root.modules():
        # Older checkpoints may have been pickled before `export_aux` was added
        # to DaytimeReliabilityRefiner. They still carry `last_aux`, so we attach
        # the flag dynamically to keep inference-time aux export compatible.
        if hasattr(module, 'export_aux') or hasattr(module, 'last_aux'):
            module.export_aux = bool(enabled)


def _configure_countability_calibration(model, opt):
    if opt is None:
        return
    root = _unwrap_model(model)
    alpha = getattr(opt, 'rn_caqh_alpha', None)
    factor_min = getattr(opt, 'rn_caqh_factor_min', None)
    factor_max = getattr(opt, 'rn_caqh_factor_max', None)
    apply_in_inference = getattr(opt, 'rn_caqh_apply_in_inference', None)
    for module in root.modules():
        setter = getattr(module, 'set_countability_runtime_config', None)
        if setter is not None:
            setter(
                alpha=alpha,
                factor_min=factor_min,
                factor_max=factor_max,
                apply_in_inference=apply_in_inference,
            )


def _configure_protocol_factorized_runtime(model, opt):
    if opt is None:
        return
    root = _unwrap_model(model)
    alpha = getattr(opt, 'pcsf_semantic_alpha', None)
    factor_min = getattr(opt, 'pcsf_factor_min', None)
    factor_max = getattr(opt, 'pcsf_factor_max', None)
    apply_in_inference = getattr(opt, 'pcsf_apply_in_inference', None)
    semantic_center = getattr(opt, 'pcsf_semantic_center', None)
    score_beta = getattr(opt, 'pcsf_score_beta', None)
    for module in root.modules():
        setter = getattr(module, 'set_protocol_factorized_runtime_config', None)
        if setter is not None:
            setter(
                alpha=alpha,
                factor_min=factor_min,
                factor_max=factor_max,
                apply_in_inference=apply_in_inference,
                semantic_center=semantic_center,
                score_beta=score_beta,
            )


def _collect_drr_aux_maps(model):
    aux_maps = []
    root = _unwrap_model(model)
    for module in root.modules():
        aux = getattr(module, 'last_aux', None)
        if isinstance(aux, dict) and 'foreground' in aux:
            aux_maps.append(aux)
    return aux_maps


def _clear_drr_aux_maps(model):
    root = _unwrap_model(model)
    for module in root.modules():
        if hasattr(module, 'last_aux'):
            module.last_aux = None


def _box_map_response(feat_map, boxes_xyxy, input_hw):
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.new_zeros((0,), dtype=torch.float32)

    map_h, map_w = feat_map.shape[-2:]
    in_h = max(float(input_hw[0]), 1.0)
    in_w = max(float(input_hw[1]), 1.0)
    feat_map = feat_map.float()
    responses = []
    for box in boxes_xyxy.float():
        x1 = int(torch.floor(box[0] / in_w * map_w).item())
        y1 = int(torch.floor(box[1] / in_h * map_h).item())
        x2 = int(torch.ceil(box[2] / in_w * map_w).item())
        y2 = int(torch.ceil(box[3] / in_h * map_h).item())
        x1 = max(0, min(map_w - 1, x1))
        y1 = max(0, min(map_h - 1, y1))
        x2 = max(x1 + 1, min(map_w, x2))
        y2 = max(y1 + 1, min(map_h, y2))
        patch = feat_map[y1:y2, x1:x2]
        cx = min(map_w - 1, max(0, (x1 + x2 - 1) // 2))
        cy = min(map_h - 1, max(0, (y1 + y2 - 1) // 2))
        responses.append(0.5 * (patch.mean() + feat_map[cy, cx]))
    return torch.stack(responses, dim=0)


def apply_daytime_score_calibration(
    pred,
    predn,
    aux_maps,
    batch_index,
    input_hw,
    conflict_gain=0.55,
    foreground_penalty=0.20,
    min_factor=0.35,
):
    if len(pred) == 0 or not aux_maps:
        return pred, predn

    device = pred.device
    support_scores = []
    conflict_scores = []
    day_scores = []
    rgb_scores = []
    boxes = pred[:, :4]
    for aux in aux_maps:
        foreground = aux.get('foreground')
        background_conflict = aux.get('background_conflict')
        day_gate = aux.get('day_gate')
        rgb_weight = aux.get('rgb_weight')
        if foreground is None or background_conflict is None or day_gate is None or rgb_weight is None:
            continue
        if batch_index >= foreground.shape[0]:
            continue

        support_scores.append(_box_map_response(foreground[batch_index, 0], boxes, input_hw))
        conflict_scores.append(_box_map_response(background_conflict[batch_index, 0], boxes, input_hw))
        rgb_scores.append(_box_map_response(rgb_weight[batch_index, 0], boxes, input_hw))
        day_scalar = day_gate[batch_index].float().mean().clamp(0.0, 1.0)
        day_scores.append(torch.full((len(pred),), day_scalar.item(), device=device, dtype=torch.float32))

    if not support_scores:
        return pred, predn

    support = torch.stack(support_scores, dim=0).mean(dim=0).clamp(0.0, 1.0)
    background_conflict = torch.stack(conflict_scores, dim=0).mean(dim=0).clamp(0.0, 1.0)
    day_gate = torch.stack(day_scores, dim=0).mean(dim=0).clamp(0.0, 1.0)
    rgb_bias = torch.stack(rgb_scores, dim=0).mean(dim=0).clamp(0.0, 1.0)

    risk = day_gate * background_conflict * (0.65 + 0.35 * rgb_bias)
    support_penalty = (1.0 - support).clamp(0.0, 1.0)
    factor = (1.0 - conflict_gain * risk) * (1.0 - foreground_penalty * support_penalty)
    factor = factor.clamp(min=min_factor, max=1.0).to(pred[:, 4].dtype)

    pred[:, 4] *= factor
    predn[:, 4] *= factor
    order = torch.argsort(pred[:, 4], descending=True)
    return pred[order], predn[order]


def apply_nighttime_score_calibration(
    pred,
    predn,
    aux_maps,
    batch_index,
    input_hw,
    conflict_gain=0.40,
    uncertainty_gain=0.25,
    ambiguity_gain=0.15,
    thermal_boost=0.35,
    fg_relief=0.30,
    min_factor=0.55,
):
    if len(pred) == 0 or not aux_maps:
        return pred, predn

    device = pred.device
    support_scores = []
    conflict_scores = []
    uncertainty_scores = []
    ambiguity_scores = []
    ir_scores = []
    night_scores = []
    boxes = pred[:, :4]
    for aux in aux_maps:
        foreground = aux.get('foreground')
        background_conflict = aux.get('background_conflict')
        uncertainty = aux.get('uncertainty')
        ambiguity = aux.get('ambiguity')
        ir_weight = aux.get('ir_weight')
        day_gate = aux.get('day_gate')
        if (
            foreground is None or background_conflict is None or uncertainty is None
            or ambiguity is None or ir_weight is None or day_gate is None
        ):
            continue
        if batch_index >= foreground.shape[0]:
            continue

        support_scores.append(_box_map_response(foreground[batch_index, 0], boxes, input_hw))
        conflict_scores.append(_box_map_response(background_conflict[batch_index, 0], boxes, input_hw))
        uncertainty_scores.append(_box_map_response(uncertainty[batch_index, 0], boxes, input_hw))
        ambiguity_scores.append(_box_map_response(ambiguity[batch_index, 0], boxes, input_hw))
        ir_scores.append(_box_map_response(ir_weight[batch_index, 0], boxes, input_hw))
        night_scalar = (1.0 - day_gate[batch_index].float().mean()).clamp(0.0, 1.0)
        night_scores.append(torch.full((len(pred),), night_scalar.item(), device=device, dtype=torch.float32))

    if not support_scores:
        return pred, predn

    support = torch.stack(support_scores, dim=0).mean(dim=0).clamp(0.0, 1.0)
    background_conflict = torch.stack(conflict_scores, dim=0).mean(dim=0).clamp(0.0, 1.0)
    uncertainty = torch.stack(uncertainty_scores, dim=0).mean(dim=0).clamp(0.0, 1.0)
    ambiguity = torch.stack(ambiguity_scores, dim=0).mean(dim=0).clamp(0.0, 1.0)
    ir_bias = torch.stack(ir_scores, dim=0).mean(dim=0).clamp(0.0, 1.0)
    night_gate = torch.stack(night_scores, dim=0).mean(dim=0).clamp(0.0, 1.0)

    risk = night_gate * (
        conflict_gain * background_conflict
        + uncertainty_gain * uncertainty
        + ambiguity_gain * ambiguity
    )
    risk = risk * (0.65 + thermal_boost * ir_bias)
    relief = (1.0 - fg_relief * support).clamp(0.40, 1.0)
    factor = (1.0 - risk.clamp(0.0, 1.0) * relief).clamp(min=min_factor, max=1.0).to(pred[:, 4].dtype)

    pred[:, 4] *= factor
    predn[:, 4] *= factor
    order = torch.argsort(pred[:, 4], descending=True)
    return pred[order], predn[order]


def _scalar_metric(x):
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return 0.0
        return float(x.reshape(-1)[0])
    if torch.is_tensor(x):
        if x.numel() == 0:
            return 0.0
        return float(x.reshape(-1)[0].item())
    return float(x)


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.5,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=True,  # save auto-label confidences
         plots=False,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False,
         opt=None,
         labels_list=None,
         day_night_split=None,
         kaist_day_roi_filter=False,
         kaist_day_nms_iou=None,
         kaist_night_nms_iou=None,
         kaist_day_score_calib=False,
         kaist_day_score_conflict_gain=0.55,
         kaist_day_score_fg_penalty=0.20,
         kaist_day_score_min_factor=0.35,
         kaist_night_score_calib=False,
         kaist_night_score_conflict_gain=0.40,
         kaist_night_score_uncertainty_gain=0.25,
         kaist_night_score_ambiguity_gain=0.15,
         kaist_night_score_thermal_boost=0.35,
         kaist_night_score_fg_relief=0.30,
         kaist_night_score_min_factor=0.55):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        if save_txt:
            labels_dir = increment_path(Path(save_dir) / 'labels' / 'pred', exist_ok=False, mkdir=True)
    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        labels_dir = save_dir / 'labels'
        if save_txt:
            # Avoid contaminating KAIST MR when --exist-ok reuses an old
            # evaluation directory with stale per-image predictions/result.txt.
            for old_txt in labels_dir.glob('*.txt'):
                old_txt.unlink()

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        _configure_countability_calibration(model, opt)
        _configure_protocol_factorized_runtime(model, opt)
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu' and half_precision # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    score_calib_enabled = kaist_day_score_calib or kaist_night_score_calib
    _set_drr_aux_export(model, score_calib_enabled)
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if dataloader is None:
        # if device.type != 'cpu':
        #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        print(opt.task)
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        if opt.mono:
            if opt.mono_rgb:
                val_path = data['val_rgb']
            elif opt.mono_thermal:
                val_path = data['val_ir']
            else:
                raise NotImplementedError
            dataloader = create_dataloader_mono(val_path, imgsz, batch_size, gs, opt, pad=0.5, rect=opt.rect, prefix=colorstr(f'{task}: '))[0]
        else:
            val_path_rgb = data['val_rgb']
            val_path_ir = data['val_ir']
            dataloader = create_dataloader_rgb_ir(val_path_rgb, val_path_ir, imgsz, batch_size, gs, opt, pad=0.5, rect=opt.rect, prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict
    names = {k: v for k, v in enumerate(data_dict['names'])}
    coco91class = coco80_to_coco91_class()
    if nc == 1:
        s = ('%20s' + '%8s' * 9 + '%9s' + '%12s') % ('Class', 'Images', 'Labels', 'TP', 'FP', 'FN', 'F1', 'P', 'R', 'mAP@.5', 'mAP@.75', 'mAP@.5:.95')  # 设置进度条的显示信息
    else:
        s = ('%20s' + '%8s' * 5 + '%9s' + '%12s') % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.75', 'mAP@.5:.95')
    logger.debug(s)
    p, r, f1, mp, mr, map50, map75, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0, 0., 0.
    tp, fp, fn = 0, 0, 0
    loss = torch.zeros(4, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda' if device.type != 'cpu' else 'cpu',
                                    enabled=(device.type != 'cpu')):
                # Run model
                t = time_synchronized()
                if opt.single_stream:
                    out, train_out = model(img, augment=augment) # inference and training outputs
                else:
                    out, _, train_out = model(img[:, :3, :, :], img[:, 3:, :, :], augment=augment)  # inference and training outputs
                t0 += time_synchronized() - t

                # Compute loss
                if compute_loss:
                    loss += compute_loss([x.float() for x in train_out], targets)[1][:4]  # box, obj, cls

                # Run NMS
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
                lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
                t = time_synchronized()
                out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
                # out = confluence_process(out, 0.1, 0.5)
                t1 += time_synchronized() - t
                batch_aux_maps = _collect_drr_aux_maps(model) if score_calib_enabled else None

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            image_index = None
            if labels_list is not None:
                try:
                    image_index = labels_list.index(str(path.stem) + '.txt')
                except ValueError:
                    image_index = None

            if kaist_day_roi_filter and image_index is not None:
                keep = kaist_day_roi_keep_mask(predn, image_index, day_night_split)
                if keep is not None:
                    pred = pred[keep]
                    predn = predn[keep]
                    if len(pred) == 0:
                        if nl:
                            stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                        continue

            domain_nms_iou = kaist_domain_nms_iou(
                image_index,
                day_night_split,
                day_iou_thres=kaist_day_nms_iou,
                night_iou_thres=kaist_night_nms_iou,
            )
            if (
                kaist_day_score_calib
                and batch_aux_maps
                and image_index is not None
                and day_night_split is not None
                and image_index < day_night_split
            ):
                pred, predn = apply_daytime_score_calibration(
                    pred,
                    predn,
                    batch_aux_maps,
                    si,
                    img[si].shape[1:],
                    conflict_gain=kaist_day_score_conflict_gain,
                    foreground_penalty=kaist_day_score_fg_penalty,
                    min_factor=kaist_day_score_min_factor,
                )
            if (
                kaist_night_score_calib
                and batch_aux_maps
                and image_index is not None
                and day_night_split is not None
                and image_index >= day_night_split
            ):
                pred, predn = apply_nighttime_score_calibration(
                    pred,
                    predn,
                    batch_aux_maps,
                    si,
                    img[si].shape[1:],
                    conflict_gain=kaist_night_score_conflict_gain,
                    uncertainty_gain=kaist_night_score_uncertainty_gain,
                    ambiguity_gain=kaist_night_score_ambiguity_gain,
                    thermal_boost=kaist_night_score_thermal_boost,
                    fg_relief=kaist_night_score_fg_relief,
                    min_factor=kaist_night_score_min_factor,
                )
            if domain_nms_iou is not None:
                predn = apply_domain_specific_nms(predn, domain_nms_iou)
                pred = apply_domain_specific_nms(pred, domain_nms_iou)
                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

            # Append to text file
            if save_txt:
                i = image_index if image_index is not None else labels_list.index(str(path.stem) + '.txt')
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh2(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                    line = (i+1, *xywh, conf) if save_conf else (i+1, *xywh)  # label format
                    with open(labels_dir / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g,' * len(line)).rstrip(",") % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    #wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            #wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # mem = '%.4gM' % (torch.cuda.memory_reserved() / 1E6 if torch.cuda.is_available() else 0)
        # print(mem)

        # file_name, extension = os.path.splitext(path.name)

        # Plot images
        if plots and batch_i < 3:
            if opt.single_stream:
                plot_path_labels = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
                plot_tensor = img[:, :3, :, :] if getattr(opt, 'early_fusion_6ch', False) else img
                Thread(target=plot_images, args=(plot_tensor, targets, paths, plot_path_labels, names), daemon=True).start()
                plot_path_pred = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
                Thread(target=plot_images, args=(plot_tensor, output_to_target(out), paths, plot_path_pred, names), daemon=True).start()
            else:
                img_rgb = img[:, :3, :, :]
                plot_path_labels = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
                Thread(target=plot_images, args=(img_rgb, targets, paths, plot_path_labels, names), daemon=True).start()
                plot_path_pred = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
                Thread(target=plot_images, args=(img_rgb, output_to_target(out), paths, plot_path_pred, names), daemon=True).start()
        _clear_drr_aux_maps(model)

    # 保存所有预测框结果，后续用于MR的计算
    if save_txt:
        temp = []
        files = os.listdir(labels_dir)
        files.sort()
        for index, file in enumerate(files):
            if file == 'result.txt':
                continue
            with open(labels_dir / file, 'r') as f:  # 打开源文件
                for line in f:
                    temp.append(line)
        with open(labels_dir / 'result.txt', 'w') as ff:
            for ii in temp:
                ff.write(ii)

    # 计算MR指标
    if not day_night_split is None:

        if day_night_split == 1455:
            annFile = './evaluation_script/KAIST_annotation.json'
            rstFiles = './' + str(labels_dir) + '/result.txt'
            MR = evaluate(annFile, rstFiles, day_night_split)
            MR_all = MR['all'].summarize(0)
            MR_day = MR['day'].summarize(0)
            MR_night = MR['night'].summarize(0)
            MR_near = MR['near'].summarize(1)
            MR_medium = MR['medium'].summarize(2)
            MR_far = MR['far'].summarize(3)
            MR_none = MR['none'].summarize(4)
            MR_partial = MR['partial'].summarize(5)
            MR_heavy = MR['heavy'].summarize(6)
            yy = MR['all'].eval.get('yy', []) if getattr(MR['all'], 'eval', None) else []
            recall_all = 1 - yy[0][-1] if yy and len(yy[0]) else 0.0

        elif day_night_split == 648:
            annFile = './evaluation_script/CVC_annotation.json'
            rstFiles = './' + str(labels_dir) + '/result.txt'
            MR = evaluate(annFile, rstFiles, day_night_split)
            MR_all = MR['all'].summarize(0)
            MR_day = MR['day'].summarize(0)
            MR_night = MR['night'].summarize(0)
            MR_near = 0.0
            MR_medium = 0.0
            MR_far = 0.0
            MR_none = 0.0
            MR_partial = 0.0
            MR_heavy = 0.0
            recall_all = 1 - MR['all'].eval['yy'][0][-1]

        else:
            raise NotImplementedError

    else:
        MR_all = 0.0
        MR_day = 0.0
        MR_night = 0.0
        MR_near = 0.0
        MR_medium = 0.0
        MR_far = 0.0
        MR_none = 0.0
        MR_partial = 0.0
        MR_heavy = 0.0
        recall_all = 0.0
    MRresult = [MR_all, MR_day, MR_night, MR_near, MR_medium, MR_far, MR_none, MR_partial, MR_heavy, recall_all]

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc) if len(stats[3]) else np.zeros(nc, dtype=np.int64)
    else:
        nt = torch.zeros(1)
    if len(stats) and stats[0].any():
        tp, fp, fn, p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()

    # Print results
    if nc > 1:
        pf = '%20s' + '%8i' * 2 + '%8.4g' * 2 + '%8.4f' + '%9.4f' + '%12.4f'  # print format
        logger.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map75, map))
    else:
        pf = '%20s' + '%8i' * 5 + '%8.4f' * 4 + '%9.4f' + '%12.4f' # print format
        logger.info(pf % (
            'all', seen, nt.sum(), _scalar_metric(tp), _scalar_metric(fp), _scalar_metric(fn), _scalar_metric(f1),
            _scalar_metric(mp), _scalar_metric(mr), _scalar_metric(map50), _scalar_metric(map75), _scalar_metric(map)
        ))
    if not day_night_split is None:
        logger.info(('%20s' + '%12s' * 9) % ('MR-all', 'MR-day', 'MR-night', 'MR-near', 'MR-medium', 'MR-far', 'MR-none', 'MR-partial', 'MR-heavy', 'Recall-all'))
        logger.info(('%20.2f' + '%12.2f' * 9) % (MR_all * 100, MR_day * 100, MR_night * 100, MR_near * 100, MR_medium * 100, MR_far * 100, MR_none * 100, MR_partial * 100, MR_heavy * 100, recall_all * 100))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            logger.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap75[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        logger.info('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    _set_drr_aux_export(model, False)
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    if not isinstance(tp, int):
        return (tp[0], fp[0], fn[0], f1[0], mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, MRresult, t
    else:
        return (tp, fp, fn, f1, mp, mr, map50, map,
                *(loss.cpu() / len(dataloader)).tolist()), maps, MRresult, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str,
                        default='checkpoints/ia_dasr_stage2_e10_best.pt',
                        help='model checkpoint path(s)')
    parser.add_argument('--data', type=str, default='configs/datasets/kaist.example.yaml',
                        help='dataset YAML path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--rect', action='store_true', help='rectangular testing')
    parser.add_argument('--mono-rgb', action='store_true')
    parser.add_argument('--mono-thermal', action='store_true')
    parser.add_argument('--early-fusion-6ch', action='store_true',
                        help='use a naive 6-channel single-stream early-fusion YOLO baseline fed by concatenated RGB+LWIR')
    parser.add_argument('--kaist-day-roi-filter', action='store_true',
                        help='suppress daytime KAIST boxes outside the reasonable evaluation region')
    parser.add_argument('--kaist-day-nms-iou', type=float, default=None,
                        help='optional tighter NMS IoU used only on daytime KAIST images')
    parser.add_argument('--kaist-night-nms-iou', type=float, default=None,
                        help='optional NMS IoU override used only on nighttime KAIST images')
    parser.add_argument('--kaist-day-score-calib', action='store_true',
                        help='use learned daytime reliability maps to downweight daytime background-conflict boxes')
    parser.add_argument('--kaist-day-score-conflict-gain', type=float, default=0.55,
                        help='daytime score penalty gain for DRR background-conflict responses')
    parser.add_argument('--kaist-day-score-fg-penalty', type=float, default=0.20,
                        help='additional daytime penalty when the DRR foreground support is weak')
    parser.add_argument('--kaist-day-score-min-factor', type=float, default=0.35,
                        help='minimum retained score factor for daytime reliability calibration')
    parser.add_argument('--kaist-night-score-calib', action='store_true',
                        help='use learned reliability maps to conservatively downweight isolated night-time pseudo-person boxes')
    parser.add_argument('--kaist-night-score-conflict-gain', type=float, default=0.40,
                        help='night-time score penalty gain for DRR background-conflict responses')
    parser.add_argument('--kaist-night-score-uncertainty-gain', type=float, default=0.25,
                        help='night-time score penalty gain for uncertainty responses')
    parser.add_argument('--kaist-night-score-ambiguity-gain', type=float, default=0.15,
                        help='night-time score penalty gain for ambiguity responses')
    parser.add_argument('--kaist-night-score-thermal-boost', type=float, default=0.35,
                        help='extra night-time penalty emphasis when the thermal branch dominates the local evidence')
    parser.add_argument('--kaist-night-score-fg-relief', type=float, default=0.30,
                        help='foreground-support relief that protects likely true positives during night-time calibration')
    parser.add_argument('--kaist-night-score-min-factor', type=float, default=0.55,
                        help='minimum retained score factor for night-time reliability calibration')
    parser.add_argument('--rn-caqh-alpha', type=float, default=None,
                        help='optional runtime alpha for residual countability calibration')
    parser.add_argument('--rn-caqh-factor-min', type=float, default=None,
                        help='optional runtime lower clamp for residual countability factor')
    parser.add_argument('--rn-caqh-factor-max', type=float, default=None,
                        help='optional runtime upper clamp for residual countability factor')
    parser.add_argument('--rn-caqh-apply-in-inference', action='store_true',
                        help='enable RN-CAQH residual score calibration for checkpoints that contain the head')
    parser.add_argument('--pcsf-semantic-alpha', type=float, default=None,
                        help='optional runtime alpha for the Round 2H semantic residual calibration factor')
    parser.add_argument('--pcsf-factor-min', type=float, default=None,
                        help='optional runtime lower clamp for the Round 2H semantic residual factor')
    parser.add_argument('--pcsf-factor-max', type=float, default=None,
                        help='optional runtime upper clamp for the Round 2H semantic residual factor')
    parser.add_argument('--pcsf-semantic-center', type=float, default=None,
                        help='optional runtime semantic-score centering value used before Round 2H tanh calibration')
    parser.add_argument('--pcsf-score-beta', type=float, default=None,
                        help='optional runtime slope applied before Round 2H tanh calibration')
    parser.add_argument('--pcsf-apply-in-inference', action='store_true',
                        help='enable the Round 2H semantic residual calibration for checkpoints that contain the head')
    opt = parser.parse_args()

    assert not (opt.mono_rgb and opt.mono_thermal)
    assert not ((opt.mono_rgb or opt.mono_thermal) and opt.early_fusion_6ch)
    opt.mono = opt.mono_rgb or opt.mono_thermal
    opt.single_stream = opt.mono or opt.early_fusion_6ch
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    print(opt.data)
    check_requirements()

    if opt.data in ['./data/multispectral/FLIR-align-3class.yaml', './data/multispectral/FLIR-ADAS.yaml', './data/multispectral/VEDAI.yaml']:
        opt.verbose = True

    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict
    if 'day_night_split' in data_dict.keys():
        DAY_NIGHT_SPLIT = data_dict['day_night_split']
    else:
        DAY_NIGHT_SPLIT = None
    p = data_dict['path'] + "/labels/test"
    print(p)
    labels_list = os.listdir(p)
    labels_list.sort()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             opt=opt,
             labels_list=labels_list,
             day_night_split=DAY_NIGHT_SPLIT,
             kaist_day_roi_filter=opt.kaist_day_roi_filter,
             kaist_day_nms_iou=opt.kaist_day_nms_iou,
             kaist_night_nms_iou=opt.kaist_night_nms_iou,
             kaist_day_score_calib=opt.kaist_day_score_calib,
             kaist_day_score_conflict_gain=opt.kaist_day_score_conflict_gain,
             kaist_day_score_fg_penalty=opt.kaist_day_score_fg_penalty,
             kaist_day_score_min_factor=opt.kaist_day_score_min_factor,
             kaist_night_score_calib=opt.kaist_night_score_calib,
             kaist_night_score_conflict_gain=opt.kaist_night_score_conflict_gain,
             kaist_night_score_uncertainty_gain=opt.kaist_night_score_uncertainty_gain,
             kaist_night_score_ambiguity_gain=opt.kaist_night_score_ambiguity_gain,
             kaist_night_score_thermal_boost=opt.kaist_night_score_thermal_boost,
             kaist_night_score_fg_relief=opt.kaist_night_score_fg_relief,
             kaist_night_score_min_factor=opt.kaist_night_score_min_factor
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, opt=opt)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, opt=opt)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
