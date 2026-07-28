import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo_test import Model as Model_multi
from models.yolo import Model as Model_mono
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader_rgb_ir, create_dataloader_mono, \
    KAIST_TYPED_AUX_GROUP, KAIST_TYPED_AUX_UNCERTAIN
from utils.general import logger, labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

from utils.datasets import RandomSampler
import global_var

import shutil


def _unwrap_model(model):
    return model.module if is_parallel(model) else model


def _configure_countability_calibration(model, opt):
    """Apply optional RN-CAQH runtime knobs without touching default Detect heads."""
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
    """Optional runtime overrides for the explicit Round 2H factorized head."""
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


def _configure_typed_semantic_runtime(model, opt):
    root = _unwrap_model(model)
    detach_semantic_probe = bool(getattr(opt, 'kaist_typed_semantic_detach', False))
    residual_scale = float(getattr(opt, 'drr_aux_typed_residual_scale', 0.0))
    residual_temperature = float(getattr(opt, 'drr_aux_typed_residual_temperature', 1.0))
    residual_group_gain = float(getattr(opt, 'drr_aux_typed_residual_group_gain', 1.0))
    residual_uncertainty_gain = float(getattr(opt, 'drr_aux_typed_residual_uncertainty_gain', 1.0))
    residual_center = float(getattr(opt, 'drr_aux_typed_residual_center', 0.50))
    countability_group_gain = float(getattr(opt, 'drr_aux_typed_countability_group_gain', 0.0))
    countability_uncertainty_gain = float(getattr(opt, 'drr_aux_typed_countability_uncertainty_gain', 0.0))
    countability_activation_floor = float(getattr(opt, 'drr_aux_typed_countability_activation_floor', 0.35))
    countability_min_gate = float(getattr(opt, 'drr_aux_typed_countability_min_gate', 0.70))
    for module in root.modules():
        setter = getattr(module, 'set_typed_semantic_runtime_config', None)
        if setter is not None:
            setter(
                detach_semantic_probe=detach_semantic_probe,
                residual_scale=residual_scale,
                residual_temperature=residual_temperature,
                residual_group_gain=residual_group_gain,
                residual_uncertainty_gain=residual_uncertainty_gain,
                residual_center=residual_center,
                countability_group_gain=countability_group_gain,
                countability_uncertainty_gain=countability_uncertainty_gain,
                countability_activation_floor=countability_activation_floor,
                countability_min_gate=countability_min_gate,
            )


def _load_partial_state_dict(model, state_dict):
    """Load matching tensors only, used for adding a new default-off research head."""
    model_state = model.state_dict()
    updated = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            updated.append(key)
    model.load_state_dict(model_state, strict=True)
    return updated, len(model_state)


def _quality_loss_weight(opt):
    countability_weight = getattr(opt, 'countability_loss_weight', None)
    if countability_weight is not None:
        return float(countability_weight)
    return float(getattr(opt, 'caqh_weight', 0.0))


def _needs_loss_ignore_targets(opt):
    return (
        bool(getattr(opt, 'kaist_ignore_aware_obj', False))
        or float(getattr(opt, 'pcsf_core_weight', 0.0)) > 0.0
        or _quality_loss_weight(opt) > 0.0
    )


def _needs_aux_ignore_targets(opt):
    return (
        float(getattr(opt, 'drr_aux_ambiguity_gain', 0.0)) > 0.0
        or float(getattr(opt, 'drr_aux_weight', 0.0)) > 0.0
        or float(getattr(opt, 'rcrcc_weight', 0.0)) > 0.0
        or float(getattr(opt, 'pcsf_factorized_weight', 0.0)) > 0.0
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


def _collect_quality_maps(model):
    root = _unwrap_model(model)
    det = getattr(root, 'model', [None])[-1]
    quality = getattr(det, 'last_quality', None)
    return quality if isinstance(quality, list) else None


def _collect_factorized_maps(model):
    root = _unwrap_model(model)
    det = getattr(root, 'model', [None])[-1]
    factorized = getattr(det, 'last_factorized', None)
    return factorized if isinstance(factorized, list) else None


def _clear_factorized_maps(model):
    root = _unwrap_model(model)
    det = getattr(root, 'model', [None])[-1]
    if det is None:
        return
    if hasattr(det, 'last_factorized'):
        det.last_factorized = None
    if hasattr(det, 'last_quality'):
        det.last_quality = None
    if hasattr(det, 'last_quality_factor'):
        det.last_quality_factor = None


def _masked_region_score(values, mask, eps=1e-6):
    if mask is None or mask.numel() == 0:
        return None, None
    mask = mask.float()
    denom = mask.sum(dim=(1, 2, 3))
    valid = denom > eps
    if not valid.any():
        return None, None
    score = (values.float() * mask).sum(dim=(1, 2, 3)) / denom.clamp_min(eps)
    return score, valid


def _region_margin_separation_loss(values, positive_mask, negative_mask, margin):
    pos_score, pos_valid = _masked_region_score(values, positive_mask)
    neg_score, neg_valid = _masked_region_score(values, negative_mask)
    if pos_score is None or neg_score is None:
        return None
    valid = pos_valid & neg_valid
    if not valid.any():
        return None
    return F.softplus(margin - (pos_score[valid] - neg_score[valid])).mean()


def _paired_region_margin_loss(pos_values, ref_values, region_mask, margin):
    pos_score, pos_valid = _masked_region_score(pos_values, region_mask)
    ref_score, ref_valid = _masked_region_score(ref_values, region_mask)
    if pos_score is None or ref_score is None:
        return None
    valid = pos_valid & ref_valid
    if not valid.any():
        return None
    return F.softplus(margin - (pos_score[valid] - ref_score[valid])).mean()


def _anchor_confidence_map(pi):
    obj = pi[..., 4].sigmoid()
    if pi.shape[-1] > 5:
        cls = pi[..., 5:].sigmoid().max(dim=-1).values
    else:
        cls = torch.ones_like(obj)
    return obj * cls


def _expand_anchor_mask(mask, anchor_count):
    if mask is None:
        return None
    if mask.dim() == 4 and mask.shape[1] == 1:
        return mask.expand(-1, anchor_count, -1, -1)
    return mask


def _gather_anchor_scores(values, b, a, gj, gi):
    if b.numel() == 0:
        return values.new_zeros((0,), dtype=torch.float32)
    b = b.long()
    a = a.long()
    gj = gj.long()
    gi = gi.long()
    if values.dim() != 4:
        return values.new_zeros((0,), dtype=torch.float32)
    _, na, ny, nx = values.shape
    if na == 1:
        a = torch.zeros_like(a)
    if ny == 1:
        gj = torch.zeros_like(gj)
    if nx == 1:
        gi = torch.zeros_like(gi)
    b, a, gj, gi, _ = _sanitize_anchor_indices_for_map(b, a, gj, gi, values)
    if b.numel() == 0:
        return values.new_zeros((0,), dtype=torch.float32)
    return values[b, a, gj, gi]


def _sanitize_anchor_indices_for_map(b, a, gj, gi, score_map):
    if b.numel() == 0:
        empty = b.new_zeros((0,), dtype=torch.long)
        return empty, empty, empty, empty, b.new_zeros((0,), dtype=torch.bool)
    _, na, ny, nx = score_map.shape
    keep = (
        (b >= 0) & (b < score_map.shape[0]) &
        (a >= 0) & (a < na) &
        (gj >= 0) & (gj < ny) &
        (gi >= 0) & (gi < nx)
    )
    if keep.all():
        return b.long(), a.long(), gj.long(), gi.long(), keep
    return b[keep].long(), a[keep].long(), gj[keep].long(), gi[keep].long(), keep


def _unique_anchor_indices(b, a, gj, gi):
    if b.numel() == 0:
        return b, a, gj, gi
    packed = torch.stack([b, a, gj, gi], dim=1)
    unique = torch.unique(packed, dim=0)
    return unique[:, 0].long(), unique[:, 1].long(), unique[:, 2].long(), unique[:, 3].long()


def _stable_unique_anchor_records(b, a, gj, gi, *extras):
    if b.numel() == 0:
        outputs = [b, a, gj, gi]
        for extra in extras:
            outputs.append(extra)
        return tuple(outputs)

    packed = torch.stack([b.long(), a.long(), gj.long(), gi.long()], dim=1).detach().cpu().tolist()
    keep = []
    seen = set()
    for idx, item in enumerate(packed):
        key = tuple(int(v) for v in item)
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    keep = torch.tensor(keep, device=b.device, dtype=torch.long)

    outputs = [b[keep].long(), a[keep].long(), gj[keep].long(), gi[keep].long()]
    for extra in extras:
        if extra is None:
            outputs.append(None)
            continue
        if isinstance(extra, torch.Tensor) and extra.shape[0] == b.shape[0]:
            outputs.append(extra[keep])
        else:
            outputs.append(extra)
    return tuple(outputs)


def _hard_negative_local_scores(score_map, ignore_mask, b, gj, gi, radius):
    if b.numel() == 0:
        return score_map.new_zeros((0,), dtype=torch.float32), score_map.new_zeros((0,), dtype=torch.bool)

    radius = int(math.ceil(max(float(radius), 0.0)))
    negatives = []
    valid_flags = []
    _, _, ny, nx = score_map.shape
    for bi, y, x in zip(b.tolist(), gj.tolist(), gi.tolist()):
        if bi < 0 or bi >= score_map.shape[0]:
            negatives.append(score_map.new_tensor(0.0))
            valid_flags.append(False)
            continue
        if y < 0 or y >= ny or x < 0 or x >= nx:
            negatives.append(score_map.new_tensor(0.0))
            valid_flags.append(False)
            continue
        y1 = max(0, y - radius)
        y2 = min(ny, y + radius + 1)
        x1 = max(0, x - radius)
        x2 = min(nx, x + radius + 1)
        local_mask = ignore_mask[bi, :, y1:y2, x1:x2].bool()
        if local_mask.any():
            local_scores = score_map[bi, :, y1:y2, x1:x2]
            negatives.append(local_scores.masked_fill(~local_mask, float('-inf')).max())
            valid_flags.append(True)
            continue

        global_mask = ignore_mask[bi].bool()
        if global_mask.any():
            negatives.append(score_map[bi].masked_fill(~global_mask, float('-inf')).max())
            valid_flags.append(True)
        else:
            negatives.append(score_map.new_tensor(0.0))
            valid_flags.append(False)

    return torch.stack(negatives, dim=0), torch.tensor(valid_flags, device=score_map.device, dtype=torch.bool)


def _sanitize_tensor(values, nan=0.0, posinf=None, neginf=None, clamp_min=None, clamp_max=None):
    if values is None:
        return None
    values = values.float()
    posinf = float(nan if posinf is None else posinf)
    neginf = float(nan if neginf is None else neginf)
    values = torch.nan_to_num(values, nan=float(nan), posinf=posinf, neginf=neginf)
    if clamp_min is not None or clamp_max is not None:
        values = values.clamp(min=clamp_min, max=clamp_max)
    return values


def _sanitize_prob(values, default=0.0):
    return _sanitize_tensor(values, nan=default, posinf=1.0, neginf=0.0, clamp_min=0.0, clamp_max=1.0)


def _safe_loss_tuple(loss, item):
    if loss is None or item is None:
        return None, None
    if not isinstance(loss, torch.Tensor):
        loss = torch.as_tensor(loss)
    if not isinstance(item, torch.Tensor):
        item = torch.as_tensor(item, device=loss.device)
    loss = loss.reshape(())
    item = item.reshape(())
    if not torch.isfinite(loss).all() or not torch.isfinite(item).all():
        return None, None
    return loss, item


def _weighted_bce_prob(prob, target, weight, eps=1e-4):
    prob = prob.float().clamp(eps, 1.0 - eps)
    target = target.float()
    weight = weight.float()
    loss = -(target * prob.log() + (1.0 - target) * (1.0 - prob).log())
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def _weighted_l1(pred, target, weight):
    pred = pred.float()
    target = target.float()
    weight = weight.float()
    return ((pred - target).abs() * weight).sum() / weight.sum().clamp_min(1.0)


def _weighted_upper_bound_loss(values, upper, weight):
    values = values.float()
    upper = upper.float()
    weight = weight.float()
    excess = (values - upper).clamp_min(0.0)
    return (excess.pow(2) * weight).sum() / weight.sum().clamp_min(1.0)


def _grouped_listwise_distill_loss(student_scores, teacher_scores, group_ids, temperature=1.0, group_weights=None):
    if student_scores is None or teacher_scores is None or group_ids is None:
        return None
    if student_scores.numel() <= 1:
        return None

    temperature = max(float(temperature), 1e-3)
    losses = []
    weights = []
    for group_id in group_ids.unique():
        mask = group_ids == group_id
        finite_mask = torch.isfinite(student_scores) & torch.isfinite(teacher_scores)
        if group_weights is not None:
            finite_mask = finite_mask & torch.isfinite(group_weights)
        mask = mask & finite_mask
        if mask.sum() <= 1:
            continue
        student_group = _sanitize_tensor(student_scores[mask], nan=0.0, posinf=1.5, neginf=0.0, clamp_min=0.0, clamp_max=1.5) / temperature
        teacher_group = _sanitize_tensor(teacher_scores[mask].detach(), nan=0.0, posinf=1.5, neginf=0.0, clamp_min=0.0, clamp_max=1.5) / temperature
        teacher_prob = F.softmax(teacher_group, dim=0).clamp_min(1e-6)
        teacher_prob = teacher_prob / teacher_prob.sum().clamp_min(1e-6)
        student_log_prob = F.log_softmax(student_group, dim=0)
        group_loss = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (temperature ** 2)
        if not torch.isfinite(group_loss).all():
            continue
        group_weight = (
            _sanitize_tensor(group_weights[mask], nan=1.0, posinf=1.0, neginf=1.0, clamp_min=0.0).mean()
            if group_weights is not None
            else group_loss.new_tensor(1.0)
        )
        if not torch.isfinite(group_weight).all():
            continue
        losses.append(group_loss)
        weights.append(group_weight)

    if not losses:
        return None

    losses = torch.stack(losses, dim=0)
    weights = torch.stack(weights, dim=0)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def _load_round2h_teacher(weights, device):
    if weights is None:
        return None
    weights = str(weights).strip()
    if len(weights) == 0 or weights.lower() == 'none':
        return None
    try:
        ckpt = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights, map_location=device)
    teacher = ckpt.get('ema', None) if isinstance(ckpt, dict) else None
    if teacher is None and isinstance(ckpt, dict):
        teacher = ckpt.get('model', None)
    if teacher is None:
        teacher = ckpt
    for module in teacher.modules():
        ensure_defaults = getattr(module, '_ensure_runtime_defaults', None)
        if callable(ensure_defaults):
            ensure_defaults()
    teacher = teacher.float().to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def _forward_round2h_teacher_outputs(teacher_model, imgs_rgb, imgs_ir):
    if teacher_model is None:
        return None
    head = getattr(teacher_model, 'model', [None])[-1]
    previous_head_training = getattr(head, 'training', False) if head is not None else False
    if head is not None:
        head.training = True
    with torch.no_grad():
        outputs = teacher_model(imgs_rgb, imgs_ir)
    if head is not None:
        head.training = previous_head_training
    teacher_model.eval()
    return outputs


def _make_soft_box_mask(targets, batch_size, height, width, device, dtype, box_pad=0.08):
    mask = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
    if targets.numel() == 0:
        return mask

    for target in targets:
        batch_index = int(target[0].item())
        if batch_index < 0 or batch_index >= batch_size:
            continue

        x, y, w, h = target[2:6].tolist()
        pad_w = w * box_pad
        pad_h = h * box_pad
        x1 = max(0, int(math.floor((x - w * 0.5 - pad_w) * width)))
        y1 = max(0, int(math.floor((y - h * 0.5 - pad_h) * height)))
        x2 = min(width, int(math.ceil((x + w * 0.5 + pad_w) * width)))
        y2 = min(height, int(math.ceil((y + h * 0.5 + pad_h) * height)))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
        mask[batch_index, 0, y1:y2, x1:x2] = 1.0

    # A small soft halo keeps far pedestrians from collapsing to a single hard
    # grid cell on low-resolution fusion maps.
    halo = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return torch.maximum(mask, halo * 0.35)


def _make_weighted_box_mask(targets, batch_size, height, width, device, dtype, box_pad=0.08, value_index=1):
    mask = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
    if targets is None or targets.numel() == 0:
        return mask

    for target in targets:
        batch_index = int(target[0].item())
        if batch_index < 0 or batch_index >= batch_size:
            continue
        weight = float(target[value_index].item()) if target.numel() > value_index else 0.0
        if weight <= 0.0:
            continue

        x, y, w, h = target[2:6].tolist()
        pad_w = w * box_pad
        pad_h = h * box_pad
        x1 = max(0, int(math.floor((x - w * 0.5 - pad_w) * width)))
        y1 = max(0, int(math.floor((y - h * 0.5 - pad_h) * height)))
        x2 = min(width, int(math.ceil((x + w * 0.5 + pad_w) * width)))
        y2 = min(height, int(math.ceil((y + h * 0.5 + pad_h) * height)))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
        current = mask[batch_index, 0, y1:y2, x1:x2]
        mask[batch_index, 0, y1:y2, x1:x2] = torch.maximum(
            current,
            current.new_full(current.shape, weight),
        )

    halo = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return torch.maximum(mask, halo * 0.35)


def _build_protocol_target_validity(targets, protocol_aux_targets):
    if targets is None or targets.numel() == 0:
        device = protocol_aux_targets.device if protocol_aux_targets is not None else 'cpu'
        return torch.zeros((0,), device=device, dtype=torch.float32)

    validity = torch.ones((targets.shape[0],), device=targets.device, dtype=torch.float32)
    if protocol_aux_targets is None or protocol_aux_targets.numel() == 0:
        return validity

    batch_ids = targets[:, 0].long().unique()
    for batch_id in batch_ids.tolist():
        target_rows = torch.where(targets[:, 0].long() == batch_id)[0]
        proto_rows = torch.where(protocol_aux_targets[:, 0].long() == batch_id)[0]
        if target_rows.numel() == 0 or proto_rows.numel() == 0:
            continue

        proto_validity = protocol_aux_targets[proto_rows, 1].float().clamp(0.0, 1.0)
        target_boxes = targets[target_rows, 2:6].float()
        proto_boxes = protocol_aux_targets[proto_rows, 2:6].float()

        if target_rows.numel() == proto_rows.numel() and torch.allclose(target_boxes, proto_boxes, atol=1e-4, rtol=1e-4):
            validity[target_rows] = proto_validity
            continue

        used_proto = torch.zeros((proto_rows.numel(),), device=targets.device, dtype=torch.bool)
        for local_idx, target_box in enumerate(target_boxes):
            distances = (proto_boxes - target_box.unsqueeze(0)).abs().sum(dim=1)
            distances = distances + used_proto.float() * 1e6
            best_idx = torch.argmin(distances)
            if used_proto[best_idx]:
                continue
            validity[target_rows[local_idx]] = proto_validity[best_idx]
            used_proto[best_idx] = True

    return validity


def _select_aux_targets_by_class(targets, class_id):
    if targets is None or targets.numel() == 0:
        return None
    class_mask = targets[:, 1].round().long() == int(class_id)
    if not class_mask.any():
        return None
    return targets[class_mask]


def _balanced_prob_bce(pred, target, eps=1e-4):
    pred = pred.float().clamp(eps, 1.0 - eps)
    target = target.float()
    loss = -(target * pred.log() + (1.0 - target) * (1.0 - pred).log())
    pos_weight = target
    neg_weight = 1.0 - target
    pos = (loss * pos_weight).sum() / pos_weight.sum().clamp_min(1.0)
    neg = (loss * neg_weight).sum() / neg_weight.sum().clamp_min(1.0)
    return 0.5 * (pos + neg)


def _prob_bce(pred, target, eps=1e-4):
    pred = pred.float().clamp(eps, 1.0 - eps)
    target = target.float()
    return -(target * pred.log() + (1.0 - target) * (1.0 - pred).log()).mean()


def _soft_dice_loss(pred, target, eps=1e-5):
    pred = pred.float()
    target = target.float()
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def _activate_aux_signal(values, floor):
    floor = min(max(float(floor), 0.0), 0.95)
    scale = max(1.0 - floor, 1e-3)
    return ((values.float() - floor).clamp(min=0.0) / scale).clamp(max=1.0)


def _find_aux_for_spatial_shape(aux_maps, height, width):
    for aux in aux_maps:
        foreground = aux.get('foreground', None)
        if foreground is not None and foreground.shape[-2:] == (height, width):
            return aux
    return None


def _compute_rcrcc_aux_loss(model, targets, ignore_targets, opt):
    rcrcc_weight = float(getattr(opt, 'rcrcc_weight', 0.0))
    if rcrcc_weight <= 0.0 or ignore_targets is None or ignore_targets.numel() == 0:
        return None, None

    aux_maps = _collect_drr_aux_maps(model)
    quality_maps = _collect_quality_maps(model)
    if not aux_maps or not quality_maps:
        return None, None

    batch_size = int(quality_maps[0].shape[0])
    if batch_size <= 0:
        return None, None

    device = targets.device if targets.numel() else ignore_targets.device
    total = torch.zeros((), device=device)

    ambiguity_gain = float(getattr(opt, 'rcrcc_ambiguity_gain', 0.50))
    conflict_gain = float(getattr(opt, 'rcrcc_conflict_gain', 0.30))
    group_gain = float(getattr(opt, 'rcrcc_group_gain', 0.70))
    uncertainty_gain = float(getattr(opt, 'rcrcc_uncertainty_gain', 0.15))
    thermal_protect = float(getattr(opt, 'rcrcc_thermal_protect', 0.35))
    risk_threshold = float(getattr(opt, 'rcrcc_risk_threshold', 0.12))
    rank_margin = float(getattr(opt, 'rcrcc_rank_margin', 0.10))
    far_target = float(getattr(opt, 'rcrcc_far_target', 0.22))
    near_target = float(getattr(opt, 'rcrcc_near_target', 0.40))
    positive_target = float(getattr(opt, 'rcrcc_positive_target', 0.78))
    positive_thermal_boost = float(getattr(opt, 'rcrcc_positive_thermal_boost', 0.12))
    positive_keep_weight = float(getattr(opt, 'rcrcc_positive_keep_weight', 0.75))
    rank_weight = float(getattr(opt, 'rcrcc_rank_weight', 0.50))
    halo_kernel = int(getattr(opt, 'rcrcc_halo_kernel', 5))
    if halo_kernel % 2 == 0:
        halo_kernel += 1
    halo_padding = halo_kernel // 2

    for quality in quality_maps:
        q_logits = _sanitize_tensor(quality.squeeze(-1), nan=0.0, posinf=8.0, neginf=-8.0, clamp_min=-8.0, clamp_max=8.0)
        _, _, h, w = q_logits.shape
        aux = _find_aux_for_spatial_shape(aux_maps, h, w)
        if aux is None:
            continue

        pos_mask = _make_soft_box_mask(targets, batch_size, h, w, device, q_logits.dtype, box_pad=0.04)
        ignore_mask = _make_soft_box_mask(ignore_targets, batch_size, h, w, device, q_logits.dtype, box_pad=0.03)
        if pos_mask is None or ignore_mask is None:
            continue

        q_prob = _sanitize_prob(q_logits.sigmoid().mean(dim=1, keepdim=True), default=0.5)

        day_gate = aux.get('day_gate', None)
        rgb_weight = aux.get('rgb_weight', None)
        ir_weight = aux.get('ir_weight', None)
        ambiguity = aux.get('ambiguity', None)
        background_conflict = aux.get('background_conflict', None)
        groupness = aux.get('groupness', None)
        uncertainty = aux.get('uncertainty', None)

        if day_gate is None or rgb_weight is None or ir_weight is None:
            continue

        ambiguity = _sanitize_prob(ambiguity, default=0.0) if ambiguity is not None else torch.zeros_like(q_prob)
        background_conflict = _sanitize_prob(background_conflict, default=0.0) if background_conflict is not None else torch.zeros_like(q_prob)
        groupness_signal = _sanitize_prob(_activate_aux_signal(groupness, 0.50), default=0.0) if groupness is not None else torch.zeros_like(q_prob)
        uncertainty_signal = _sanitize_prob(_activate_aux_signal(uncertainty, 0.55), default=0.0) if uncertainty is not None else torch.zeros_like(q_prob)
        day_gate = _sanitize_prob(day_gate, default=0.0)
        rgb_weight = _sanitize_prob(rgb_weight, default=0.5)
        ir_weight = _sanitize_prob(ir_weight, default=0.5)

        semantic_risk = (
            ambiguity_gain * ambiguity
            + conflict_gain * background_conflict
            + group_gain * groupness_signal
            + uncertainty_gain * uncertainty_signal
        ) / max(ambiguity_gain + conflict_gain + group_gain + uncertainty_gain, 1e-3)

        risk = (
            day_gate.float()
            * semantic_risk.clamp(0.0, 1.0)
            * (0.65 + 0.35 * rgb_weight.float())
            * (1.0 - thermal_protect * ir_weight.float()).clamp(min=0.55, max=1.0)
        ).clamp(0.0, 1.0)

        pos_halo = F.max_pool2d(pos_mask, kernel_size=halo_kernel, stride=1, padding=halo_padding)
        ignore_clean = (ignore_mask * (1.0 - pos_mask)).clamp_min(0.0)
        risky_ignore = ignore_clean * risk.detach()
        local_ignore = ignore_clean * pos_halo * (risk.detach() >= risk_threshold).float()
        if local_ignore.sum() <= 1.0:
            local_ignore = ignore_clean * (risk.detach() >= risk_threshold).float()

        layer_loss = torch.zeros((), device=device)

        if risky_ignore.sum() > 1.0:
            upper = torch.full_like(q_prob, far_target)
            upper[pos_halo > 0.0] = near_target
            ignore_excess = (q_prob - upper).clamp_min(0.0)
            layer_loss = layer_loss + (ignore_excess.pow(2) * risky_ignore).sum() / risky_ignore.sum().clamp_min(1.0)

        if pos_mask.sum() > 1.0:
            pos_floor = (positive_target + positive_thermal_boost * ir_weight.float()).clamp(max=0.95)
            pos_keep = ((pos_floor - q_prob).clamp_min(0.0) * pos_mask).sum() / pos_mask.sum().clamp_min(1.0)
            layer_loss = layer_loss + positive_keep_weight * pos_keep

            margin_loss = _region_margin_separation_loss(q_prob, pos_mask.detach(), local_ignore.detach(), rank_margin)
            if margin_loss is not None:
                layer_loss = layer_loss + rank_weight * margin_loss

        total = total + layer_loss

    total = total / max(len(quality_maps), 1)
    if not torch.isfinite(total).all():
        return None, None
    scaled = total * rcrcc_weight * batch_size
    if not torch.isfinite(scaled).all():
        return None, None
    return scaled.reshape(()), (total.detach() * rcrcc_weight).reshape(())


def _compute_round2h_factorized_loss(model, teacher_model, pred, targets, ignore_targets, protocol_aux_targets, imgs_rgb, imgs_ir, compute_loss, opt):
    factorized_weight = float(getattr(opt, 'pcsf_factorized_weight', 0.0))
    if factorized_weight <= 0.0:
        return None, None

    factorized_maps = _collect_factorized_maps(model)
    if not factorized_maps:
        return None, None

    aux_maps = _collect_drr_aux_maps(model)
    if not aux_maps:
        return None, None

    device = pred[0].device
    batch_size = int(pred[0].shape[0])
    if batch_size <= 0:
        return None, None

    targets = targets if targets is not None else torch.zeros((0, 6), device=device)
    ignore_targets = ignore_targets if ignore_targets is not None else torch.zeros((0, 6), device=device)
    protocol_aux_targets = protocol_aux_targets if protocol_aux_targets is not None else torch.zeros((0, 6), device=device)

    humanness_weight = float(getattr(opt, 'pcsf_h_weight', 1.0))
    countability_weight = float(getattr(opt, 'pcsf_c_weight', 1.0))
    groupness_weight = float(getattr(opt, 'pcsf_g_weight', 0.35))
    uncertainty_weight = float(getattr(opt, 'pcsf_u_weight', 0.20))
    bias_weight = float(getattr(opt, 'pcsf_bias_weight', 0.35))
    pair_rank_weight = float(getattr(opt, 'pcsf_pair_rank_weight', 0.70))
    semantic_rank_weight = float(getattr(opt, 'pcsf_semantic_rank_weight', 0.35))
    pair_rank_margin = float(getattr(opt, 'pcsf_pair_rank_margin', 0.10))
    pair_radius = float(getattr(opt, 'pcsf_pair_radius', 2.0))
    ignore_target = float(getattr(opt, 'pcsf_ignore_target', 0.25))
    background_weight = float(getattr(opt, 'pcsf_background_weight', 0.03))
    group_floor = float(getattr(opt, 'pcsf_group_floor', 0.50))
    uncertainty_floor = float(getattr(opt, 'pcsf_uncertainty_floor', 0.55))
    alpha_order_weight = float(getattr(opt, 'pcsf_alpha_order_weight', 0.10))
    alpha_order_margin = float(getattr(opt, 'pcsf_alpha_order_margin', 0.05))
    teacher_distill_weight = float(getattr(opt, 'pcsf_teacher_distill_weight', 0.0))
    teacher_keep_margin = float(getattr(opt, 'pcsf_teacher_keep_margin', 0.02))
    teacher_night_boost = float(getattr(opt, 'pcsf_teacher_night_boost', 0.30))
    teacher_listwise_weight = float(getattr(opt, 'pcsf_teacher_listwise_weight', 0.0))
    teacher_listwise_temperature = float(getattr(opt, 'pcsf_teacher_listwise_temperature', 1.0))
    bias_day_gain = float(getattr(opt, 'pcsf_bias_day_gain', 0.45))
    bias_conflict_gain = float(getattr(opt, 'pcsf_bias_conflict_gain', 0.35))
    bias_group_gain = float(getattr(opt, 'pcsf_bias_group_gain', 0.35))
    bias_density_gain = float(getattr(opt, 'pcsf_bias_density_gain', 0.20))
    bias_thermal_protect = float(getattr(opt, 'pcsf_bias_thermal_protect', 0.35))
    bias_ambiguity_gain = float(getattr(opt, 'pcsf_bias_ambiguity_gain', 0.25))
    amb_upper_weight = float(getattr(opt, 'pcsf_amb_upper_weight', 0.0))
    amb_far_target = float(getattr(opt, 'pcsf_amb_far_target', ignore_target))
    amb_near_target = float(getattr(opt, 'pcsf_amb_near_target', max(ignore_target, 0.40)))
    amb_near_radius = float(getattr(opt, 'pcsf_amb_near_radius', 2.0))
    protocol_weight_floor = float(getattr(opt, 'pcsf_protocol_weight_floor', 0.25))
    protocol_rank_min_validity = float(getattr(opt, 'pcsf_protocol_rank_min_validity', 0.15))
    protocol_distill_min_validity = float(getattr(opt, 'pcsf_protocol_distill_min_validity', 0.20))
    protocol_invalid_gain = float(getattr(opt, 'pcsf_protocol_invalid_gain', 0.35))
    duplicate_halo_weight = float(getattr(opt, 'pcsf_duplicate_halo_weight', 0.0))
    duplicate_halo_radius = float(getattr(opt, 'pcsf_duplicate_halo_radius', 0.0))
    duplicate_halo_target = float(getattr(opt, 'pcsf_duplicate_halo_target', 0.22))
    night_isolated_weight = float(getattr(opt, 'pcsf_night_isolated_weight', 0.0))
    night_isolated_upper = float(getattr(opt, 'pcsf_night_isolated_upper', 0.18))
    night_isolated_radius = float(getattr(opt, 'pcsf_night_isolated_radius', 0.0))
    night_isolated_min_risk = float(getattr(opt, 'pcsf_night_isolated_min_risk', 0.12))
    night_isolated_conflict_gain = float(getattr(opt, 'pcsf_night_isolated_conflict_gain', 0.50))
    night_isolated_uncertainty_gain = float(getattr(opt, 'pcsf_night_isolated_uncertainty_gain', 0.30))
    night_isolated_ambiguity_gain = float(getattr(opt, 'pcsf_night_isolated_ambiguity_gain', 0.20))
    ignore_scale = float(getattr(opt, 'kaist_ignore_core_scale', 1.0))
    ignore_pad = float(getattr(opt, 'kaist_ignore_box_pad', 0.03))
    protocol_weight_floor = min(max(protocol_weight_floor, 0.0), 1.0)
    protocol_rank_min_validity = min(max(protocol_rank_min_validity, 0.0), 1.0)
    protocol_distill_min_validity = min(max(protocol_distill_min_validity, 0.0), 1.0)
    protocol_invalid_gain = min(max(protocol_invalid_gain, 0.0), 1.0)
    duplicate_halo_weight = max(duplicate_halo_weight, 0.0)
    duplicate_halo_radius = max(duplicate_halo_radius, 0.0)
    duplicate_halo_target = min(max(duplicate_halo_target, 0.0), 1.0)
    night_isolated_weight = max(night_isolated_weight, 0.0)
    night_isolated_upper = min(max(night_isolated_upper, 0.0), 1.0)
    night_isolated_radius = max(night_isolated_radius, 0.0)
    night_isolated_min_risk = min(max(night_isolated_min_risk, 0.0), 1.0)
    night_isolated_conflict_gain = max(night_isolated_conflict_gain, 0.0)
    night_isolated_uncertainty_gain = max(night_isolated_uncertainty_gain, 0.0)
    night_isolated_ambiguity_gain = max(night_isolated_ambiguity_gain, 0.0)

    teacher_pred = None
    if teacher_model is not None and (teacher_distill_weight > 0.0 or teacher_listwise_weight > 0.0):
        teacher_pred = _forward_round2h_teacher_outputs(teacher_model, imgs_rgb, imgs_ir)

    total = torch.zeros((), device=device)
    used_layers = 0

    tcls, tbox, indices, anchors, offsets, target_ids = compute_loss.build_targets(pred, targets)
    protocol_target_validity = _build_protocol_target_validity(targets, protocol_aux_targets)
    protocol_invalid_targets = None
    if protocol_aux_targets is not None and protocol_aux_targets.numel() > 0:
        protocol_invalid_targets = protocol_aux_targets.clone()
        protocol_invalid_targets[:, 1] = (1.0 - protocol_invalid_targets[:, 1]).clamp(0.0, 1.0)

    for layer_idx, info in enumerate(factorized_maps):
        if layer_idx >= len(pred):
            continue
        pi = pred[layer_idx]
        _, na, h, w, _ = pi.shape
        aux = _find_aux_for_spatial_shape(aux_maps, h, w)
        if aux is None:
            continue

        humanness = _sanitize_prob(info['humanness'], default=0.5).squeeze(-1)
        countability = _sanitize_prob(info['countability'], default=0.5).squeeze(-1)
        groupness = _sanitize_prob(info['groupness'], default=0.0).squeeze(-1)
        uncertainty = _sanitize_prob(info['uncertainty'], default=0.0).squeeze(-1)
        bias_term = _sanitize_tensor(info['bias_term'], nan=0.0, posinf=4.0, neginf=-4.0, clamp_min=-4.0, clamp_max=4.0).squeeze(-1)
        semantic = _sanitize_prob(info['semantic'], default=0.5).squeeze(-1)
        factor = _sanitize_tensor(info['factor'], nan=1.0, posinf=1.0, neginf=1.0, clamp_min=0.0, clamp_max=2.0).squeeze(-1)
        base_score = _sanitize_prob(_anchor_confidence_map(pi), default=0.0)
        final_score = _sanitize_tensor(base_score * factor, nan=0.0, posinf=2.0, neginf=0.0, clamp_min=0.0, clamp_max=2.0)

        pos_mask = _make_soft_box_mask(targets, batch_size, h, w, device, humanness.dtype, box_pad=0.04)
        ignore_mask = _make_soft_box_mask(ignore_targets, batch_size, h, w, device, humanness.dtype, box_pad=0.03)
        protocol_valid_mask = _make_weighted_box_mask(protocol_aux_targets, batch_size, h, w, device, humanness.dtype, box_pad=0.04)
        if protocol_valid_mask.sum() <= 0.0:
            protocol_valid_mask = pos_mask.float()
        protocol_invalid_mask = _make_weighted_box_mask(protocol_invalid_targets, batch_size, h, w, device, humanness.dtype, box_pad=0.04)
        if pos_mask is None or ignore_mask is None:
            continue

        pos_anchor = _expand_anchor_mask(pos_mask, na)
        ignore_anchor = _expand_anchor_mask(ignore_mask, na)
        protocol_valid_anchor = _expand_anchor_mask(protocol_valid_mask.clamp(0.0, 1.0), na)
        ignore_only = (ignore_anchor * (1.0 - pos_anchor)).clamp(0.0, 1.0)
        human_target = torch.maximum(pos_anchor, ignore_only)
        countability_target = (protocol_valid_anchor + ignore_only * ignore_target).clamp(0.0, 1.0)
        background = (1.0 - torch.maximum(pos_anchor, ignore_anchor)).clamp(0.0, 1.0)

        human_weight_map = background_weight * background + human_target
        positive_count_weight = pos_anchor * (
            protocol_weight_floor + (1.0 - protocol_weight_floor) * protocol_valid_anchor.detach()
        )
        countability_weight_map = background_weight * background + positive_count_weight + ignore_only

        layer_loss = torch.zeros((), device=device)
        if humanness_weight > 0.0:
            layer_loss = layer_loss + humanness_weight * _weighted_bce_prob(humanness, human_target, human_weight_map)
        if countability_weight > 0.0:
            layer_loss = layer_loss + countability_weight * _weighted_bce_prob(countability, countability_target, countability_weight_map)

        foreground = _sanitize_prob(aux.get('foreground', None), default=0.0)
        day_gate = _sanitize_prob(aux.get('day_gate', None), default=0.0)
        rgb_weight = _sanitize_prob(aux.get('rgb_weight', None), default=0.5)
        ir_weight = _sanitize_prob(aux.get('ir_weight', None), default=0.5)
        background_conflict = _sanitize_prob(aux.get('background_conflict', None), default=0.0)
        aux_groupness = _sanitize_prob(aux.get('groupness', None), default=0.0)
        aux_uncertainty = _sanitize_prob(aux.get('uncertainty', None), default=0.0)
        ambiguity = _sanitize_prob(aux.get('ambiguity', None), default=0.0)

        if foreground is None or day_gate is None or rgb_weight is None or ir_weight is None or background_conflict is None:
            continue

        group_target = _activate_aux_signal(aux_groupness, group_floor) if aux_groupness is not None else torch.zeros_like(pos_mask)
        uncertainty_target = _activate_aux_signal(aux_uncertainty, uncertainty_floor) if aux_uncertainty is not None else torch.zeros_like(pos_mask)
        semantic_context_weight = (
            ignore_mask
            + protocol_invalid_gain * protocol_invalid_mask
            + 0.25 * pos_mask
            + 0.10 * foreground.float()
        ).clamp(max=1.0)
        semantic_context_weight = _expand_anchor_mask(semantic_context_weight, na)

        if groupness_weight > 0.0:
            layer_loss = layer_loss + groupness_weight * _weighted_bce_prob(
                groupness,
                _expand_anchor_mask(group_target, na),
                semantic_context_weight,
            )
        if uncertainty_weight > 0.0:
            layer_loss = layer_loss + uncertainty_weight * _weighted_bce_prob(
                uncertainty,
                _expand_anchor_mask(uncertainty_target, na),
                semantic_context_weight,
            )

        structured_risk = None
        if bias_weight > 0.0 or amb_upper_weight > 0.0:
            density = F.avg_pool2d(
                (ignore_mask + protocol_invalid_gain * protocol_invalid_mask + 0.50 * pos_mask).clamp(max=1.0),
                kernel_size=5,
                stride=1,
                padding=2,
            )
            ambiguity_target = ambiguity.float() if ambiguity is not None else ignore_mask
            ambiguity_target = torch.maximum(ambiguity_target, protocol_invalid_mask.clamp(0.0, 1.0))
            structured_risk = day_gate.float() * (
                bias_day_gain * (0.50 + 0.50 * rgb_weight.float())
                + bias_conflict_gain * background_conflict.float()
                + bias_group_gain * group_target.float()
                + bias_density_gain * density.float()
                + bias_ambiguity_gain * ambiguity_target.float()
            )
            protect = bias_thermal_protect * ir_weight.float()
            bias_target = (structured_risk - protect).clamp(0.0, 1.0) * 2.0 - 1.0
            bias_supervision = (ignore_mask + protocol_invalid_mask + 0.25 * pos_mask + 0.05 * background).clamp(max=1.0)
            if bias_weight > 0.0:
                layer_loss = layer_loss + bias_weight * _weighted_l1(
                    bias_term,
                    _expand_anchor_mask(bias_target, na),
                    _expand_anchor_mask(bias_supervision, na),
                )

        layer_target_ids = target_ids[layer_idx].long()
        b, a, gj, gi = indices[layer_idx]
        b = b.long()
        a = a.long()
        gj = gj.long()
        gi = gi.long()
        raw_b, raw_a, raw_gj, raw_gi, raw_keep = _sanitize_anchor_indices_for_map(b, a, gj, gi, final_score)
        layer_target_ids = layer_target_ids[raw_keep]
        raw_target_ids = layer_target_ids
        raw_b_for_near, raw_gj_for_near, raw_gi_for_near = raw_b, raw_gj, raw_gi
        b, a, gj, gi, layer_target_ids = _stable_unique_anchor_records(raw_b, raw_a, raw_gj, raw_gi, layer_target_ids)
        pos_validity = (
            protocol_target_validity[layer_target_ids].float().clamp(0.0, 1.0)
            if layer_target_ids.numel() > 0 and protocol_target_validity.numel() > 0
            else final_score.new_ones((b.numel(),), dtype=torch.float32)
        )
        positive_anchor_bool = torch.zeros_like(final_score, dtype=torch.bool)
        if b.numel() > 0:
            positive_anchor_bool[b, a, gj, gi] = True
        protocol_invalid_anchor_bool = None
        if protocol_invalid_targets is not None and protocol_invalid_targets.numel() > 0:
            protocol_invalid_anchor_bool = compute_loss.build_ignore_mask(
                pi,
                protocol_invalid_targets,
                box_scale=1.0,
                box_pad=0.04,
            )

        if amb_upper_weight > 0.0 and ignore_targets.numel() > 0:
            ignore_anchor_bool = compute_loss.build_ignore_mask(
                pi,
                ignore_targets,
                box_scale=ignore_scale,
                box_pad=ignore_pad,
            )
            near_pos_mask = compute_loss.build_positive_proximity_mask(
                pi,
                raw_b_for_near,
                raw_gj_for_near,
                raw_gi_for_near,
                radius=amb_near_radius,
            )
            ignore_anchor_bool = ignore_anchor_bool & (~positive_anchor_bool)
            if ignore_anchor_bool.any():
                risk_anchor = (
                    _expand_anchor_mask(structured_risk.clamp(0.0, 1.0), na)
                    if structured_risk is not None
                    else ignore_anchor_bool.float()
                )
                upper = torch.full_like(countability, amb_far_target)
                upper[near_pos_mask] = amb_near_target
                upper_loss = _weighted_upper_bound_loss(
                    countability,
                    upper,
                    ignore_anchor_bool.float() * risk_anchor,
                )
                layer_loss = layer_loss + amb_upper_weight * upper_loss

        if duplicate_halo_weight > 0.0 and b.numel() > 0:
            duplicate_halo_mask = compute_loss.build_positive_proximity_mask(
                pi,
                raw_b_for_near,
                raw_gj_for_near,
                raw_gi_for_near,
                radius=duplicate_halo_radius,
            )
            duplicate_halo_mask = duplicate_halo_mask & (~positive_anchor_bool)
            if duplicate_halo_mask.any():
                if ignore_targets.numel() > 0:
                    ignore_anchor_bool = compute_loss.build_ignore_mask(
                        pi,
                        ignore_targets,
                        box_scale=ignore_scale,
                        box_pad=ignore_pad,
                    )
                    duplicate_halo_mask = duplicate_halo_mask & (~ignore_anchor_bool)
                if protocol_invalid_anchor_bool is not None:
                    duplicate_halo_mask = duplicate_halo_mask & (~protocol_invalid_anchor_bool)
                if duplicate_halo_mask.any():
                    duplicate_density = F.avg_pool2d(
                        (pos_mask + 0.50 * ignore_mask + protocol_invalid_gain * protocol_invalid_mask).clamp(max=1.0),
                        kernel_size=5,
                        stride=1,
                        padding=2,
                    )
                    crowd_context = torch.maximum(group_target.float(), duplicate_density.float()).clamp(0.0, 1.0)
                    duplicate_weight = duplicate_halo_mask.float() * _expand_anchor_mask(crowd_context, na)
                    if duplicate_weight.sum() > 0.0:
                        duplicate_upper = torch.full_like(final_score, duplicate_halo_target)
                        duplicate_loss = _weighted_upper_bound_loss(final_score, duplicate_upper, duplicate_weight)
                        layer_loss = layer_loss + duplicate_halo_weight * duplicate_loss

        if night_isolated_weight > 0.0:
            night_candidate_mask = torch.ones_like(final_score, dtype=torch.bool)
            if night_isolated_radius > 0.0 and raw_b_for_near.numel() > 0:
                near_pos_mask = compute_loss.build_positive_proximity_mask(
                    pi,
                    raw_b_for_near,
                    raw_gj_for_near,
                    raw_gi_for_near,
                    radius=night_isolated_radius,
                )
                night_candidate_mask = night_candidate_mask & (~near_pos_mask)
            night_candidate_mask = night_candidate_mask & (~positive_anchor_bool)
            if ignore_targets.numel() > 0:
                ignore_anchor_bool = compute_loss.build_ignore_mask(
                    pi,
                    ignore_targets,
                    box_scale=ignore_scale,
                    box_pad=ignore_pad,
                )
                night_candidate_mask = night_candidate_mask & (~ignore_anchor_bool)
            if protocol_invalid_anchor_bool is not None:
                night_candidate_mask = night_candidate_mask & (~protocol_invalid_anchor_bool)
            if night_candidate_mask.any():
                night_gate = (1.0 - day_gate.float()).clamp(0.0, 1.0)
                night_risk = night_gate * (
                    night_isolated_conflict_gain * background_conflict.float()
                    + night_isolated_uncertainty_gain * uncertainty_target.float()
                    + night_isolated_ambiguity_gain * ambiguity.float()
                )
                night_risk = night_risk * (0.65 + 0.35 * ir_weight.float())
                night_risk = night_risk * (1.0 - 0.50 * foreground.float()).clamp(0.25, 1.0)
                night_risk = night_risk.clamp(0.0, 1.0)
                night_weight = night_candidate_mask.float() * _expand_anchor_mask(night_risk, na)
                night_weight = torch.where(
                    night_weight >= night_isolated_min_risk,
                    night_weight,
                    torch.zeros_like(night_weight),
                )
                if night_weight.sum() > 0.0:
                    night_upper = torch.full_like(final_score, night_isolated_upper)
                    night_loss = _weighted_upper_bound_loss(final_score, night_upper, night_weight)
                    layer_loss = layer_loss + night_isolated_weight * night_loss

        if b.numel() > 0 and ignore_targets.numel() > 0 and (pair_rank_weight > 0.0 or semantic_rank_weight > 0.0):
            ignore_anchor_bool = compute_loss.build_ignore_mask(
                pi,
                ignore_targets,
                box_scale=ignore_scale,
                box_pad=ignore_pad,
            )
            ignore_anchor_bool = ignore_anchor_bool & (~positive_anchor_bool)

            pos_final = _gather_anchor_scores(final_score, b, a, gj, gi)
            pos_semantic = _gather_anchor_scores(semantic, b, a, gj, gi)
            neg_final, valid_final = _hard_negative_local_scores(final_score, ignore_anchor_bool, b, gj, gi, pair_radius)
            neg_semantic, valid_sem = _hard_negative_local_scores(semantic, ignore_anchor_bool, b, gj, gi, pair_radius)
            pos_day = _gather_anchor_scores(
                _expand_anchor_mask(day_gate.float(), na),
                b, a, gj, gi
            )
            pair_weight = (1.0 + 0.50 * pos_day) * (
                protocol_weight_floor + (1.0 - protocol_weight_floor) * pos_validity
            )

            valid_final_pair = valid_final & (pos_validity >= protocol_rank_min_validity)
            valid_sem_pair = valid_sem & (pos_validity >= protocol_rank_min_validity)

            if pair_rank_weight > 0.0 and valid_final_pair.any():
                pair_loss = F.softplus(pair_rank_margin - (pos_final[valid_final_pair] - neg_final[valid_final_pair]))
                layer_loss = layer_loss + pair_rank_weight * (
                    pair_loss * pair_weight[valid_final_pair]
                ).sum() / pair_weight[valid_final_pair].sum().clamp_min(1.0)

            if semantic_rank_weight > 0.0 and valid_sem_pair.any():
                sem_loss = F.softplus(pair_rank_margin - (pos_semantic[valid_sem_pair] - neg_semantic[valid_sem_pair]))
                layer_loss = layer_loss + semantic_rank_weight * (
                    sem_loss * pair_weight[valid_sem_pair]
                ).sum() / pair_weight[valid_sem_pair].sum().clamp_min(1.0)

            if teacher_pred is not None and layer_idx < len(teacher_pred):
                teacher_score = _anchor_confidence_map(teacher_pred[layer_idx]).float().detach()
                teacher_pos = _gather_anchor_scores(teacher_score, b, a, gj, gi)
                thermal_weight = _gather_anchor_scores(
                    _expand_anchor_mask(ir_weight.float(), na),
                    b, a, gj, gi
                )
                keep_weight = (1.0 + teacher_night_boost * thermal_weight) * (
                    protocol_weight_floor + (1.0 - protocol_weight_floor) * pos_validity
                )
                teacher_keep_mask = pos_validity >= protocol_distill_min_validity
                if teacher_keep_mask.any():
                    distill_reg = F.smooth_l1_loss(pos_final[teacher_keep_mask], teacher_pos[teacher_keep_mask], reduction='none')
                    keep_loss = F.softplus(
                        teacher_keep_margin - (pos_final[teacher_keep_mask] - teacher_pos[teacher_keep_mask])
                    )
                    teacher_loss = (distill_reg + keep_loss) * keep_weight[teacher_keep_mask]
                    layer_loss = layer_loss + teacher_distill_weight * (
                        teacher_loss.sum() / keep_weight[teacher_keep_mask].sum().clamp_min(1.0)
                    )
                if teacher_listwise_weight > 0.0 and teacher_keep_mask.sum() > 1:
                    listwise_loss = _grouped_listwise_distill_loss(
                        pos_final[teacher_keep_mask],
                        teacher_pos[teacher_keep_mask],
                        b[teacher_keep_mask],
                        temperature=teacher_listwise_temperature,
                        group_weights=keep_weight[teacher_keep_mask],
                    )
                    if listwise_loss is not None:
                        layer_loss = layer_loss + teacher_listwise_weight * listwise_loss

        total = total + layer_loss
        used_layers += 1

    if used_layers == 0:
        return None, None

    total = total / used_layers
    if not torch.isfinite(total).all():
        return None, None
    root = _unwrap_model(model)
    det = getattr(root, 'model', [None])[-1]
    if alpha_order_weight > 0.0 and hasattr(det, 'alpha_g_raw') and hasattr(det, 'alpha_u_raw'):
        alpha_g = F.softplus(det.alpha_g_raw)
        alpha_u = F.softplus(det.alpha_u_raw)
        total = total + alpha_order_weight * F.softplus(alpha_u - alpha_g + alpha_order_margin)
        if not torch.isfinite(total).all():
            return None, None

    scaled = total * factorized_weight * batch_size
    if not torch.isfinite(scaled).all():
        return None, None
    return scaled.reshape(()), (total.detach() * factorized_weight).reshape(())


def _compute_drr_aux_loss(model, targets, ignore_targets, typed_aux_targets, imgs_rgb, opt, epoch, batch_i, nb, start_epoch):
    aux_weight = float(getattr(opt, 'drr_aux_weight', 0.0))
    if aux_weight <= 0.0 or imgs_rgb is None:
        return None, None

    aux_maps = _collect_drr_aux_maps(model)
    if not aux_maps:
        return None, None

    batch_size = imgs_rgb.shape[0]
    device = imgs_rgb.device
    total = imgs_rgb.new_tensor(0.0)
    fg_gain = float(getattr(opt, 'drr_aux_fg_gain', 1.0))
    conflict_gain = float(getattr(opt, 'drr_aux_conflict_gain', 0.35))
    rgb_bg_gain = float(getattr(opt, 'drr_aux_rgb_bg_gain', 0.15))
    day_gain = float(getattr(opt, 'drr_aux_day_gain', 0.25))
    ambiguity_gain = float(getattr(opt, 'drr_aux_ambiguity_gain', 0.0))
    group_gain = float(getattr(opt, 'drr_aux_group_gain', 0.0))
    uncertainty_gain = float(getattr(opt, 'drr_aux_uncertainty_gain', 0.0))
    typed_consistency_gain = float(getattr(opt, 'drr_aux_typed_consistency_gain', 0.0))
    box_pad = float(getattr(opt, 'drr_aux_box_pad', 0.08))
    ignore_box_pad = float(getattr(opt, 'drr_aux_ignore_box_pad', 0.04))
    typed_box_pad = float(getattr(opt, 'drr_aux_typed_box_pad', ignore_box_pad))

    for aux in aux_maps:
        foreground = _sanitize_prob(aux['foreground'], default=0.0)
        _, _, h, w = foreground.shape
        target_mask = _make_soft_box_mask(targets, batch_size, h, w, device, foreground.dtype, box_pad)

        fg_loss = _balanced_prob_bce(foreground.float(), target_mask.float())
        fg_loss = fg_loss + 0.5 * _soft_dice_loss(foreground, target_mask)

        background_conflict = aux.get('background_conflict', None)
        conflict_loss = foreground.new_tensor(0.0)
        rgb_bg_loss = foreground.new_tensor(0.0)
        if background_conflict is not None:
            background_conflict = _sanitize_prob(background_conflict, default=0.0)
            # Do not suppress true pedestrian regions: this protects the already
            # strong night detections and crowded true positives.
            conflict_loss = (background_conflict.float() * target_mask.float()).sum() / target_mask.float().sum().clamp_min(1.0)

            # Daytime false positives are often RGB-texture/background-conflict
            # responses. Penalize only that unreliable conjunction outside boxes.
            if 'rgb_weight' in aux and 'day_gate' in aux:
                bg_mask = (1.0 - target_mask.float()).detach()
                day_gate = _sanitize_prob(aux['day_gate'], default=0.0)
                rgb_bg_loss = (
                    _sanitize_prob(aux['rgb_weight'], default=0.5)
                    * background_conflict.float()
                    * day_gate
                    * bg_mask
                ).sum() / bg_mask.sum().clamp_min(1.0)

        ambiguity_loss = foreground.new_tensor(0.0)
        ignore_mask = None
        if ambiguity_gain > 0.0 and 'ambiguity' in aux and ignore_targets is not None and ignore_targets.numel():
            ignore_mask = _make_soft_box_mask(ignore_targets, batch_size, h, w, device, foreground.dtype, ignore_box_pad)
            # Countable pedestrians and ambiguous human-like regions are mutually exclusive
            # for the reasonable-person protocol. The overlap guard prevents noisy boxes
            # from teaching valid pedestrians to disappear.
            ambiguity_target = (ignore_mask * (1.0 - target_mask)).detach()
            ambiguity = _sanitize_prob(aux['ambiguity'], default=0.0)
            ambiguity_loss = _balanced_prob_bce(ambiguity, ambiguity_target.float())
            ambiguity_loss = ambiguity_loss + 0.35 * (
                ambiguity * target_mask.float()
            ).sum() / target_mask.float().sum().clamp_min(1.0)

        groupness_loss = foreground.new_tensor(0.0)
        uncertainty_loss = foreground.new_tensor(0.0)
        typed_consistency_loss = foreground.new_tensor(0.0)
        typed_union_mask = None
        if ignore_mask is None and ignore_targets is not None and ignore_targets.numel():
            ignore_mask = _make_soft_box_mask(ignore_targets, batch_size, h, w, device, foreground.dtype, ignore_box_pad)
        if typed_aux_targets is not None and typed_aux_targets.numel():
            group_targets = _select_aux_targets_by_class(typed_aux_targets, KAIST_TYPED_AUX_GROUP)
            uncertainty_targets = _select_aux_targets_by_class(typed_aux_targets, KAIST_TYPED_AUX_UNCERTAIN)
            group_mask = None
            uncertainty_mask = None
            if group_targets is not None and group_targets.numel():
                group_mask = _make_soft_box_mask(
                    group_targets,
                    batch_size,
                    h,
                    w,
                    device,
                    foreground.dtype,
                    typed_box_pad,
                )
            if uncertainty_targets is not None and uncertainty_targets.numel():
                uncertainty_mask = _make_soft_box_mask(
                    uncertainty_targets,
                    batch_size,
                    h,
                    w,
                    device,
                    foreground.dtype,
                    typed_box_pad,
                )
            if group_mask is not None or uncertainty_mask is not None:
                typed_union_mask = torch.zeros_like(target_mask)
                if group_mask is not None:
                    typed_union_mask = torch.maximum(typed_union_mask, group_mask)
                if uncertainty_mask is not None:
                    typed_union_mask = torch.maximum(typed_union_mask, uncertainty_mask)

            negative_mask = torch.ones_like(target_mask)
            negative_mask = negative_mask - target_mask
            if ignore_mask is not None:
                negative_mask = negative_mask - ignore_mask
            if typed_union_mask is not None:
                negative_mask = negative_mask - typed_union_mask
            negative_mask = negative_mask.clamp_min(0.0).detach()
            typed_rank_margin = float(getattr(opt, 'drr_aux_typed_rank_margin', 0.15))
            typed_consistency_margin = float(getattr(opt, 'drr_aux_typed_consistency_margin', 0.05))

            if group_gain > 0.0 and 'groupness' in aux and group_mask is not None:
                groupness = _sanitize_prob(aux['groupness'], default=0.0)
                loss = _region_margin_separation_loss(groupness, group_mask, negative_mask, typed_rank_margin)
                if loss is not None:
                    groupness_loss = loss

            if uncertainty_gain > 0.0 and 'uncertainty' in aux and uncertainty_mask is not None:
                uncertainty = _sanitize_prob(aux['uncertainty'], default=0.0)
                loss = _region_margin_separation_loss(uncertainty, uncertainty_mask, negative_mask, typed_rank_margin)
                if loss is not None:
                    uncertainty_loss = loss

            if typed_consistency_gain > 0.0 and 'ambiguity' in aux:
                ambiguity = _sanitize_prob(aux['ambiguity'], default=0.0)
                consistency_terms = []
                if group_mask is not None and 'groupness' in aux:
                    loss = _paired_region_margin_loss(ambiguity, _sanitize_prob(aux['groupness'], default=0.0), group_mask, typed_consistency_margin)
                    if loss is not None:
                        consistency_terms.append(loss)
                if uncertainty_mask is not None and 'uncertainty' in aux:
                    loss = _paired_region_margin_loss(ambiguity, _sanitize_prob(aux['uncertainty'], default=0.0), uncertainty_mask, typed_consistency_margin)
                    if loss is not None:
                        consistency_terms.append(loss)
                if consistency_terms:
                    typed_consistency_loss = torch.stack(consistency_terms, dim=0).mean()

        total = (
            total
            + fg_gain * fg_loss
            + conflict_gain * conflict_loss
            + rgb_bg_gain * rgb_bg_loss
            + ambiguity_gain * ambiguity_loss
            + group_gain * groupness_loss
            + uncertainty_gain * uncertainty_loss
            + typed_consistency_gain * typed_consistency_loss
        )

    total = total / max(len(aux_maps), 1)
    if not torch.isfinite(total).all():
        return None, None

    if day_gain > 0.0:
        # Weak scene supervision from the augmented visible image brightness.
        # It only teaches the router when daytime-style suppression should be
        # available; detection labels still decide the final boxes.
        brightness = imgs_rgb.detach().float().mean(dim=(1, 2, 3))
        threshold = float(getattr(opt, 'drr_aux_day_thr', 0.28))
        tau = max(float(getattr(opt, 'drr_aux_day_tau', 0.08)), 1e-3)
        day_target = torch.sigmoid((brightness - threshold) / tau).clamp(0.02, 0.98)
        day_preds = []
        for aux in aux_maps:
            if 'day_gate' in aux:
                day_preds.append(aux['day_gate'].float().view(batch_size, -1).mean(dim=1))
        if day_preds:
            day_pred = torch.stack(day_preds, dim=0).mean(dim=0).clamp(1e-4, 1.0 - 1e-4)
            total = total + day_gain * _prob_bce(day_pred, day_target)
            if not torch.isfinite(total).all():
                return None, None

    warmup = float(getattr(opt, 'drr_aux_warmup', 2.0))
    if warmup > 0.0:
        # Use absolute training epoch progress instead of local continuation
        # progress. Otherwise, fine-tuning from a converged checkpoint will
        # incorrectly re-warm the DRR auxiliary path after restart, which can
        # temporarily weaken the very supervision that stabilizes late-stage
        # FP suppression.
        progress = (epoch + (batch_i + 1) / max(nb, 1)) / warmup
        aux_scale = max(0.0, min(1.0, progress))
    else:
        aux_scale = 1.0

    scaled = total * aux_weight * aux_scale * batch_size
    if not torch.isfinite(scaled).all():
        return None, None
    return scaled, (total.detach() * aux_weight * aux_scale)


def train_rgb_ir(hyp, opt, device, tb_writer=None):
    os.environ["WANDB_MODE"] = "offline"
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Save source
    try:
        shutil.copy('./models/common.py', save_dir / 'common.py') # copy common.py
        shutil.copy(opt.cfg, save_dir / 'net.yaml') # copy net architecture
    except: # resume
        print("Resume checkpoint.")

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(seed=1 + rank, deterministic=True)
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        if weights.endswith('.pt') and os.path.isfile(weights):
            try:
                run_id = torch.load(weights, weights_only=False).get('wandb_id')
            except TypeError:
                run_id = torch.load(weights).get('wandb_id')
        else:
            run_id = None
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming


    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    Model = Model_mono if opt.single_stream else Model_multi
    pretrained = weights.endswith('.pt')
    #pretrained = False
    if pretrained:
        if not opt.resume: # from pre-trained yolo weights
            with torch_distributed_zero_first(rank):
                attempt_download(weights)  # download if not found locally
            try:
                ckpt = torch.load(weights, map_location=device, weights_only=False)  # load checkpoint
            except TypeError:
                ckpt = torch.load(weights, map_location=device)
            input_ch = 6 if getattr(opt, 'early_fusion_6ch', False) else 3
            model = Model(opt.cfg or ckpt['model'].yaml, ch=input_ch, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

            state_dict = ckpt['model'].float().state_dict()  # to FP32
            new_state_dict = model.state_dict()
            updated = []

            backbone = Path(weights).stem # weights should name after */yolo**.pt
            if backbone in ['yolov5s', 'yolov5n', 'yolov5m', 'yolov5l', 'yolov5']:
                if backbone in ['yolov5s', 'yolov5n', 'yolov5m', 'yolov5l']:
                    backbone_depth = 10
                    yolo_depth = 25
                else:
                    backbone_depth = 5
                    yolo_depth = 20

                tiers = list(set([int(key.split('.')[1]) for key in list(new_state_dict.keys())]))
                new_model_depth = max(tiers)+1 # total depth of given model; assume {(backbone: vis branch, ir branch); fuse; head}
                expand = new_model_depth - yolo_depth if not opt.single_stream else 0
                logger.debug("backbone: %s; b_depth: %d; t_depth: %d; f_depth: %d" %(backbone, backbone_depth, yolo_depth, expand-backbone_depth))
                logger.debug("transfered model depth: %d, vis_depth: %d, ir_depth: %d, fuse_depth: %d" %(new_model_depth, backbone_depth, backbone_depth, expand-backbone_depth))

                for key in list(state_dict.keys()):
                    tier = int(key.split('.')[1])

                    if tier < backbone_depth: # backbone
                        if key in new_state_dict:
                            if new_state_dict[key].shape == state_dict[key].shape: # copy pre-trained weights to vis branch
                                new_state_dict[key] = state_dict[key]
                                updated.append(key)
                                logger.debug("%s: copied from %s" %(key, key))
                            else:
                                logger.debug("%s: shape mismatch: %s and %s" %(key, str(new_state_dict[key].shape), str(state_dict[key].shape)))
                        else:
                            logger.debug("%s: not in model" %key)

                        if not opt.single_stream: # dual branch model: copy pretrained weights to another branch
                            new_key = 'model.' + str(tier+backbone_depth) + key[7:]
                            if new_key in new_state_dict:
                                if new_state_dict[new_key].shape == state_dict[key].shape: # copy pre-trained weights to ir branch
                                    new_state_dict[new_key] = state_dict[key]
                                    updated.append(new_key)
                                    logger.debug("%s: copied from %s" %(new_key, key))
                                else:
                                    logger.debug("%s: shape mismatch: %s and %s" %(key, str(new_state_dict[new_key].shape), str(state_dict[key].shape)))
                            else:
                                logger.debug("%s: not in model" %new_key)

                    elif tier < yolo_depth-1: # head (detect not included)
                        if opt.single_stream:
                            new_key = key
                        else:
                            new_key = 'model.' + str(tier+expand) + key[8:]
                            logger.debug("setting new key = %s" %new_key)
                        if new_key in new_state_dict:
                            if new_state_dict[new_key].shape == state_dict[key].shape: # copy pre-trained weights to head
                                new_state_dict[new_key] = state_dict[key]
                                updated.append(new_key)
                                logger.debug("%s: copied from %s" %(new_key, key))
                            else:
                                logger.debug("%s: shape mismatch: %s and %s" %(key, str(new_state_dict[new_key].shape), str(state_dict[key].shape)))
                        else:
                            logger.debug("%s: not in model" %new_key)

                    else: # detect
                        logger.debug("%s: skipped" %key)
                        pass
            else:
                for key, value in state_dict.items():
                    if key in new_state_dict and new_state_dict[key].shape == value.shape:
                        new_state_dict[key] = value
                        updated.append(key)
                        logger.debug("%s: copied by exact partial transfer" % key)

            for key in list(new_state_dict.keys()):
                if not key in updated:
                    logger.debug("%s: not updated" %key)

            model.load_state_dict(new_state_dict, strict=True)  # load
            logger.info('Transferred %g/%g items from %s' % (len(updated), len(new_state_dict), weights))  # report

        else: # from fine-tuned weights
            try:
                ckpt = torch.load(weights, map_location=device, weights_only=False) # load checkpoint
            except TypeError:
                ckpt = torch.load(weights, map_location=device)
            input_ch = 6 if getattr(opt, 'early_fusion_6ch', False) else 3
            model = Model(opt.cfg or ckpt['model'].yaml, ch=input_ch, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            if getattr(opt, 'allow_partial_load', False):
                updated, total = _load_partial_state_dict(model, state_dict)
                logger.info('Partially transferred %g/%g items from %s' % (len(updated), total, weights))
            else:
                model.load_state_dict(state_dict, strict=True) # load
                logger.info('Resumed from pre-trained %s' %weights)

    else:
        input_ch = 6 if getattr(opt, 'early_fusion_6ch', False) else 3
        model = Model(opt.cfg, ch=input_ch, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    _configure_countability_calibration(model, opt)
    _configure_protocol_factorized_runtime(model, opt)
    _configure_typed_semantic_runtime(model, opt)

    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    if opt.mono:
        if opt.mono_rgb:
            train_path = data_dict['train_rgb']
            test_path = data_dict['val_rgb']
        elif opt.mono_thermal:
            train_path = data_dict['train_ir']
            test_path = data_dict['val_ir']
        else:
            raise NotImplementedError
    else:
        train_path_rgb = data_dict['train_rgb']
        test_path_rgb = data_dict['val_rgb']
        train_path_ir = data_dict['train_ir']
        test_path_ir = data_dict['val_ir']
    labels_path = os.path.join(data_dict['path'], 'labels', 'test')
    if 'day_night_split' in data_dict.keys():
        DAY_NIGHT_SPLIT = data_dict['day_night_split']
    else:
        DAY_NIGHT_SPLIT = None
    labels_list = os.listdir(labels_path)
    labels_list.sort()

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    optimized_params = []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
            optimized_params.append(k+'.bias')
        if isinstance(v, nn.BatchNorm2d) or isinstance(v, nn.LayerNorm) or isinstance(v, nn.GroupNorm):
            pg0.append(v.weight)  # no decay
            optimized_params.append(k+'.weight')
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
            optimized_params.append(k+'.weight')
    for k, v in model.named_parameters():
        if not (k in optimized_params):
            pg2.append(v) # learnable parameters (as biases)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(pg1)} weight, {len(pg0)} weight (no decay), {len(pg2)} bias")
    del pg0, pg1, pg2

    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    # print("nl", nl)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    if opt.mono:
        dataloader, dataset = create_dataloader_mono(train_path, imgsz, batch_size, gs, opt,
                                                     hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                                     world_size=opt.world_size, workers=opt.workers,
                                                     image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    else:
        dataloader, dataset = create_dataloader_rgb_ir(train_path_rgb, train_path_ir, imgsz, batch_size, gs, opt,
                                                       hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                                       world_size=opt.world_size, workers=opt.workers,
                                                       image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        if opt.mono:
            testloader, testdata = create_dataloader_mono(test_path, imgsz_test, 1, gs, opt,
                                                          hyp=hyp, cache=opt.cache_images and not opt.notest, rect=opt.rect,
                                                          rank=-1, world_size=opt.world_size, workers=opt.workers,
                                                          pad=0.5, prefix=colorstr('val: '))
        else:
            testloader, testdata = create_dataloader_rgb_ir(test_path_rgb, test_path_ir, imgsz_test, 1, gs, opt,
                                                            hyp=hyp, cache=opt.cache_images and not opt.notest, rect=opt.rect,
                                                            rank=-1, world_size=opt.world_size, workers=opt.workers,
                                                            pad=0.5, prefix=colorstr('val: '))

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    MRresult = 0.0
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    amp_enabled = cuda and not getattr(opt, 'no_amp', False)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    warned_nonfinite_drr = False
    warned_nonfinite_rcrcc = False
    warned_nonfinite_round2h = False
    warned_nonfinite_total = False
    compute_loss = ComputeLoss(model, opt)  # init loss class
    teacher_model = _load_round2h_teacher(getattr(opt, 'pcsf_teacher_weights', None), device)
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'AMP mixed precision: {"enabled" if amp_enabled else "disabled"}\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    raw_last = wdir / 'last_raw.pt'
    raw_best = wdir / 'best_raw.pt'

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        # for k, v in model.named_parameters():
        #     if v.requires_grad == False:
        #         print(k)
        # exit()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'aux/rank', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()

        for i, batch in pbar:  # batch -------------------------------------------------------------
            if len(batch) == 7:
                imgs, targets, ignore_targets, typed_aux_targets, protocol_aux_targets, paths, _ = batch
            elif len(batch) == 6:
                imgs, targets, ignore_targets, typed_aux_targets, paths, _ = batch
                protocol_aux_targets = None
            elif len(batch) == 5:
                imgs, targets, ignore_targets, paths, _ = batch
                typed_aux_targets = None
                protocol_aux_targets = None
            else:
                imgs, targets, paths, _ = batch
                ignore_targets = None
                typed_aux_targets = None
                protocol_aux_targets = None
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            if opt.single_stream:
                # FQY my code 训练数据可视化
                flage_visual = global_var.get_value('flag_visual_training_dataset')
                if flage_visual:
                    from torchvision import transforms
                    unloader = transforms.ToPILImage()
                    for num in range(batch_size):
                        image = imgs[num, :3, :, :].cpu().clone() if getattr(opt, 'early_fusion_6ch', False) else imgs[num, :, :, :].cpu().clone()  # clone the tensor
                        image = image.squeeze(0)  # remove the fake batch dimension
                        image = unloader(image)
                        image.save('example_%s_%s_%s.jpg'%(str(epoch), str(i), str(num)))

            else:
                imgs_rgb = imgs[:, :3, :, :]
                imgs_ir = imgs[:, 3:, :, :]

                # FQY my code 训练数据可视化
                flage_visual = global_var.get_value('flag_visual_training_dataset')
                if flage_visual:
                    from torchvision import transforms
                    unloader = transforms.ToPILImage()
                    for num in range(batch_size):
                        image = imgs[num, :3, :, :].cpu().clone()  # clone the tensor
                        image = image.squeeze(0)  # remove the fake batch dimension
                        image = unloader(image)
                        image.save('example_%s_%s_%s_color.jpg'%(str(epoch), str(i), str(num)))
                        image = imgs[num, 3:, :, :].cpu().clone()  # clone the tensor
                        image = image.squeeze(0)  # remove the fake batch dimension
                        image = unloader(image)
                        image.save('example_%s_%s_%s_ir.jpg'%(str(epoch), str(i), str(num)))

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with torch.amp.autocast(device_type='cuda' if cuda else 'cpu', enabled=amp_enabled):
                if opt.single_stream:
                    pred = model(imgs)  # forward
                    targets_device = targets.to(device)
                    loss, loss_items = compute_loss(pred, targets_device)  # loss scaled by batch_size
                else:
                    pred = model(imgs_rgb, imgs_ir)  # forward
                    targets_device = targets.to(device)
                    ignore_targets_device = (
                        ignore_targets.to(device)
                        if ignore_targets is not None and (
                            _needs_loss_ignore_targets(opt) or _needs_aux_ignore_targets(opt)
                        )
                        else None
                    )
                    ignore_targets_for_loss = (
                        ignore_targets_device
                        if ignore_targets_device is not None and _needs_loss_ignore_targets(opt)
                        else None
                    )
                    typed_aux_targets_device = (
                        typed_aux_targets.to(device)
                        if typed_aux_targets is not None and getattr(opt, 'kaist_typed_aux', False)
                        else None
                    )
                    protocol_aux_targets_device = (
                        protocol_aux_targets.to(device)
                        if protocol_aux_targets is not None and getattr(opt, 'kaist_protocol_aux', False)
                        else None
                    )
                    loss, loss_items = compute_loss(pred, targets_device, ignore_targets_for_loss)  # loss scaled by batch_size
                    drr_aux_loss, drr_aux_item = _compute_drr_aux_loss(
                        model, targets_device, ignore_targets_device, typed_aux_targets_device, imgs_rgb, opt, epoch, i, nb, start_epoch
                    )
                    raw_drr_aux_loss, raw_drr_aux_item = drr_aux_loss, drr_aux_item
                    drr_aux_loss, drr_aux_item = _safe_loss_tuple(drr_aux_loss, drr_aux_item)
                    if raw_drr_aux_loss is not None and drr_aux_loss is None and rank in [-1, 0] and not warned_nonfinite_drr:
                        print(f'WARNING: skipped non-finite DRR auxiliary loss at epoch {epoch}, batch {i}.')
                        warned_nonfinite_drr = True
                    if drr_aux_loss is not None:
                        loss = loss + drr_aux_loss
                        loss_items = loss_items.clone()
                        loss_items[3] += drr_aux_item.detach()
                    rcrcc_aux_loss, rcrcc_aux_item = _compute_rcrcc_aux_loss(
                        model, targets_device, ignore_targets_device, opt
                    )
                    raw_rcrcc_aux_loss, raw_rcrcc_aux_item = rcrcc_aux_loss, rcrcc_aux_item
                    rcrcc_aux_loss, rcrcc_aux_item = _safe_loss_tuple(rcrcc_aux_loss, rcrcc_aux_item)
                    if raw_rcrcc_aux_loss is not None and rcrcc_aux_loss is None and rank in [-1, 0] and not warned_nonfinite_rcrcc:
                        print(f'WARNING: skipped non-finite RCRCC auxiliary loss at epoch {epoch}, batch {i}.')
                        warned_nonfinite_rcrcc = True
                    if rcrcc_aux_loss is not None:
                        loss = loss + rcrcc_aux_loss
                        if not isinstance(loss_items, torch.Tensor):
                            loss_items = torch.tensor(loss_items, device=loss.device)
                        else:
                            loss_items = loss_items.clone()
                        loss_items[3] += rcrcc_aux_item.detach().reshape(())
                    round2h_aux_loss, round2h_aux_item = _compute_round2h_factorized_loss(
                        model,
                        teacher_model,
                        pred,
                        targets_device,
                        ignore_targets_device,
                        protocol_aux_targets_device,
                        imgs_rgb,
                        imgs_ir,
                        compute_loss,
                        opt,
                    )
                    raw_round2h_aux_loss, raw_round2h_aux_item = round2h_aux_loss, round2h_aux_item
                    round2h_aux_loss, round2h_aux_item = _safe_loss_tuple(round2h_aux_loss, round2h_aux_item)
                    if raw_round2h_aux_loss is not None and round2h_aux_loss is None and rank in [-1, 0] and not warned_nonfinite_round2h:
                        print(f'WARNING: skipped non-finite Round 2H/2I auxiliary loss at epoch {epoch}, batch {i}.')
                        warned_nonfinite_round2h = True
                    if round2h_aux_loss is not None:
                        loss = loss + round2h_aux_loss
                        if not isinstance(loss_items, torch.Tensor):
                            loss_items = torch.tensor(loss_items, device=loss.device)
                        else:
                            loss_items = loss_items.clone()
                        loss_items[3] += round2h_aux_item.detach().reshape(())
                _clear_drr_aux_maps(model)
                _clear_factorized_maps(model)
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.
                if not torch.isfinite(loss).all():
                    if rank in [-1, 0] and not warned_nonfinite_total:
                        print(f'WARNING: skipped non-finite total loss at epoch {epoch}, batch {i}.')
                        warned_nonfinite_total = True
                    optimizer.zero_grad()
                    continue

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                grad_clip_norm = float(getattr(opt, 'grad_clip_norm', 0.0) or 0.0)
                grad_value_clip = float(getattr(opt, 'grad_value_clip', 0.0) or 0.0)
                if grad_clip_norm > 0.0 or grad_value_clip > 0.0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    if grad_value_clip > 0.0:
                        for param in model.parameters():
                            if param.grad is None:
                                continue
                            param.grad.data = torch.nan_to_num(
                                param.grad.data,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )
                            param.grad.data.clamp_(min=-grad_value_clip, max=grad_value_clip)
                if grad_clip_norm > 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    if not torch.isfinite(grad_norm).all():
                        if rank in [-1, 0]:
                            print(f'WARNING: skipped optimizer step with non-finite gradients at epoch {epoch}, batch {i}.')
                        optimizer.zero_grad()
                        continue
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                if ni < 3:
                    plot_targets = targets.detach().cpu()
                    if opt.single_stream:
                        f = save_dir / f'train_batch{ni}.jpg'
                        plot_tensor = imgs[:, :3, :, :] if getattr(opt, 'early_fusion_6ch', False) else imgs
                        plot_tensor = plot_tensor.detach().cpu()
                        Thread(target=plot_images, args=(plot_tensor, plot_targets, paths, f), daemon=True).start()
                    else:
                        f1 = save_dir / f'train_batch{ni}_vis.jpg'
                        f2 = save_dir / f'train_batch{ni}_inf.jpg'
                        plot_rgb = imgs_rgb.detach().cpu()
                        plot_ir = imgs_ir.detach().cpu()
                        Thread(target=plot_images, args=(plot_rgb, plot_targets, paths, f1), daemon=True).start()
                        Thread(target=plot_images, args=(plot_ir, plot_targets, paths, f2), daemon=True).start()

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, MRresult, times = test.test(data_dict,
                                                           batch_size=1,
                                                           imgsz=imgsz_test,
                                                           model=ema.ema,
                                                           single_cls=opt.single_cls,
                                                           dataloader=testloader,
                                                           save_dir=save_dir,
                                                           save_txt=True,
                                                           save_conf=True,
                                                           verbose=nc < 50 and final_epoch,
                                                           plots=plots and final_epoch,
                                                           wandb_logger=wandb_logger,
                                                           compute_loss=compute_loss,
                                                           is_coco=is_coco,
                                                           opt=opt,
                                                           labels_list=labels_list,
                                                           day_night_split=DAY_NIGHT_SPLIT,
                                                           kaist_day_roi_filter=getattr(opt, 'kaist_day_roi_filter', False),
                                                           kaist_day_nms_iou=getattr(opt, 'kaist_day_nms_iou', None),
                                                           kaist_night_nms_iou=getattr(opt, 'kaist_night_nms_iou', None),
                                                           kaist_day_score_calib=getattr(opt, 'kaist_day_score_calib', False),
                                                           kaist_day_score_conflict_gain=getattr(opt, 'kaist_day_score_conflict_gain', 0.55),
                                                           kaist_day_score_fg_penalty=getattr(opt, 'kaist_day_score_fg_penalty', 0.20),
                                                           kaist_day_score_min_factor=getattr(opt, 'kaist_day_score_min_factor', 0.35),
                                                           kaist_night_score_calib=getattr(opt, 'kaist_night_score_calib', False),
                                                           kaist_night_score_conflict_gain=getattr(opt, 'kaist_night_score_conflict_gain', 0.40),
                                                           kaist_night_score_uncertainty_gain=getattr(opt, 'kaist_night_score_uncertainty_gain', 0.25),
                                                           kaist_night_score_ambiguity_gain=getattr(opt, 'kaist_night_score_ambiguity_gain', 0.15),
                                                           kaist_night_score_thermal_boost=getattr(opt, 'kaist_night_score_thermal_boost', 0.35),
                                                           kaist_night_score_fg_relief=getattr(opt, 'kaist_night_score_fg_relief', 0.30),
                                                           kaist_night_score_min_factor=getattr(opt, 'kaist_night_score_min_factor', 0.55)
                                                           )

            # log
            keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/aux_or_rank_loss',  # train loss
                    'TP', 'FP', 'FN', 'F1', 'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # metrics
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss', 'val/rank_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2',  # learning rate
                    'MR_all', 'MR_day', 'MR_night', 'MR_near', 'MR_medium', 'MR_far', 'MR_none', 'MR_partial', 'MR_heavy', 'Recall_all'  # MR
                    ]
            vals = list(mloss) + list(results) + lr + MRresult
            dicts = {k: v for k, v in zip(keys, vals)}  # dict
            file = save_dir / 'results.csv'
            n = len(dicts) + 1  # number of cols
            s = '' if file.exists() else (('%s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

            # Update best mAP
            if not DAY_NIGHT_SPLIT is None: # CVC or KAIST
                fi = 100 - MRresult[0] # pick (100 - MR_all)
            else: # FLIR or LLVIP
                fi = fitness(np.array(results).reshape(1, -1), os.path.basename(opt.data).lower())  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            #wandb_logger.end_epoch(best_result=best_fitness == fi)
            # fi = MRresult[0]
            # if fi < best_fitness:
            #     best_fitness = fi

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if getattr(opt, 'save_raw_best_last', False):
                    ckpt_raw = {
                        'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(model.module if is_parallel(model) else model).float(),
                        'ema': deepcopy(ema.ema).float(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}
                    torch.save(ckpt_raw, raw_last)
                    if best_fitness == fi:
                        torch.save(ckpt_raw, raw_best)
                    del ckpt_raw
                if ((epoch + 1) % opt.save_period == 0) and opt.save_period != -1:
                    torch.save(ckpt, wdir / f'epoch{epoch}.pt')
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    t1 = time.time()
    t = t1 - t0
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(file=save_dir / 'results.csv')  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        for m in (last, best) if best.exists() else (last,):  # speed, mAP tests
            results, _, MRresult, _ = test.test(opt.data,
                                                batch_size=1,
                                                imgsz=imgsz_test,
                                                conf_thres=0.001,
                                                iou_thres=0.5,
                                                model=attempt_load(m, device),
                                                single_cls=opt.single_cls,
                                                dataloader=testloader,
                                                save_dir=save_dir,
                                                save_txt=True,
                                                save_conf=True,
                                                save_json=False,
                                                plots=False,
                                                is_coco=is_coco,
                                                labels_list=labels_list,
                                                verbose=nc > 1,
                                                opt=opt,
                                                day_night_split=DAY_NIGHT_SPLIT,
                                                kaist_day_roi_filter=getattr(opt, 'kaist_day_roi_filter', False),
                                                kaist_day_nms_iou=getattr(opt, 'kaist_day_nms_iou', None),
                                                kaist_night_nms_iou=getattr(opt, 'kaist_night_nms_iou', None),
                                                kaist_day_score_calib=getattr(opt, 'kaist_day_score_calib', False),
                                                kaist_day_score_conflict_gain=getattr(opt, 'kaist_day_score_conflict_gain', 0.55),
                                                kaist_day_score_fg_penalty=getattr(opt, 'kaist_day_score_fg_penalty', 0.20),
                                                kaist_day_score_min_factor=getattr(opt, 'kaist_day_score_min_factor', 0.35),
                                                kaist_night_score_calib=getattr(opt, 'kaist_night_score_calib', False),
                                                kaist_night_score_conflict_gain=getattr(opt, 'kaist_night_score_conflict_gain', 0.40),
                                                kaist_night_score_uncertainty_gain=getattr(opt, 'kaist_night_score_uncertainty_gain', 0.25),
                                                kaist_night_score_ambiguity_gain=getattr(opt, 'kaist_night_score_ambiguity_gain', 0.15),
                                                kaist_night_score_thermal_boost=getattr(opt, 'kaist_night_score_thermal_boost', 0.35),
                                                kaist_night_score_fg_relief=getattr(opt, 'kaist_night_score_fg_relief', 0.30),
                                                kaist_night_score_min_factor=getattr(opt, 'kaist_night_score_min_factor', 0.55)
                                                )

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='',
                        help='optional initial weights path; empty starts from scratch')
    parser.add_argument('--cfg', type=str, default='configs/models/ia_dasr_formal.yaml',
                        help='model YAML path')
    parser.add_argument('--data', type=str, default='configs/datasets/kaist.example.yaml',
                        help='dataset YAML path')
    parser.add_argument('--hyp', type=str, default='configs/experiments/hyp.formal_stage2.yaml',
                        help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--no-amp', action='store_true',
                        help='disable CUDA automatic mixed precision for numerically sensitive fine-tuning')
    parser.add_argument('--grad-clip-norm', type=float, default=0.0,
                        help='clip gradient global norm before optimizer.step; 0 disables clipping')
    parser.add_argument('--grad-value-clip', type=float, default=0.0,
                        help='clip individual gradient values after finite sanitization; 0 disables value clipping')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--save-raw-best-last', action='store_true',
                        help='also save non-stripped FP32 last_raw.pt and best_raw.pt for stable paper-facing evaluation')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--mono-rgb', action='store_true')
    parser.add_argument('--mono-thermal', action='store_true')
    parser.add_argument('--early-fusion-6ch', action='store_true',
                        help='use a naive 6-channel single-stream early-fusion YOLO baseline fed by concatenated RGB+LWIR')
    parser.add_argument('--kaist-day-roi-filter', action='store_true',
                        help='use KAIST daytime valid-region calibration during validation/testing')
    parser.add_argument('--kaist-day-nms-iou', type=float, default=None,
                        help='optional tighter NMS IoU used only on daytime KAIST validation/testing images')
    parser.add_argument('--kaist-night-nms-iou', type=float, default=None,
                        help='optional NMS IoU override used only on nighttime KAIST validation/testing images')
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
    parser.add_argument('--crowd-repgt-weight', type=float, default=0.0,
                        help='repulsion loss weight for suppressing overlap with nearby non-target GT boxes')
    parser.add_argument('--crowd-repbox-weight', type=float, default=0.0,
                        help='repulsion loss weight for suppressing duplicate boxes assigned to different GTs')
    parser.add_argument('--crowd-rep-sigma', type=float, default=0.5,
                        help='smooth-ln transition used by the crowd repulsion loss')
    parser.add_argument('--crowd-rep-iou-thr', type=float, default=0.0,
                        help='minimum overlap required before a crowd repulsion pair contributes to the loss')
    parser.add_argument('--kaist-ignore-aware-obj', action='store_true',
                        help='mask objectness background loss inside KAIST ambiguous human regions during training')
    parser.add_argument('--kaist-typed-aux', action='store_true',
                        help='load typed ambiguous KAIST labels from raw sanitized annotations for auxiliary semantic supervision')
    parser.add_argument('--kaist-typed-annotation-root', type=str, default=None,
                        help='directory containing raw sanitized KAIST training annotation txt files')
    parser.add_argument('--kaist-protocol-aux', action='store_true',
                        help='load raw person annotations with soft reasonable-protocol validity targets for Round 2I completion')
    parser.add_argument('--kaist-protocol-annotation-root', type=str, default=None,
                        help='directory containing raw KAIST training annotation txt files used to derive soft protocol validity')
    parser.add_argument('--kaist-protocol-height-thr', type=float, default=55.0,
                        help='reasonable-protocol minimum height threshold used to form soft protocol validity')
    parser.add_argument('--kaist-protocol-height-tau', type=float, default=8.0,
                        help='softness of the height-validity transition around the reasonable-protocol threshold')
    parser.add_argument('--kaist-protocol-heavy-occ-value', type=float, default=0.25,
                        help='soft protocol validity assigned to heavily occluded raw person boxes')
    parser.add_argument('--kaist-protocol-boundary-left', type=float, default=5.0,
                        help='left reasonable-evaluation boundary used for protocol validity')
    parser.add_argument('--kaist-protocol-boundary-top', type=float, default=5.0,
                        help='top reasonable-evaluation boundary used for protocol validity')
    parser.add_argument('--kaist-protocol-boundary-right', type=float, default=635.0,
                        help='right reasonable-evaluation boundary used for protocol validity')
    parser.add_argument('--kaist-protocol-boundary-bottom', type=float, default=507.0,
                        help='bottom reasonable-evaluation boundary used for protocol validity')
    parser.add_argument('--kaist-protocol-validity-combine', type=str, default='min',
                        choices=['min', 'product', 'geomean'],
                        help='soft logical AND used to combine size, occlusion, and boundary protocol-validity components')
    parser.add_argument('--kaist-protocol-validity-floor', type=float, default=0.0,
                        help='optional minimum retained protocol validity used to avoid fully silent auxiliary targets')
    parser.add_argument('--kaist-ignore-core-scale', type=float, default=1.0,
                        help='scale of the fully ignored core inside ambiguous KAIST boxes; <1 exposes boundaries')
    parser.add_argument('--kaist-ignore-border-scale', type=float, default=1.0,
                        help='scale of the weakly weighted ignore-border band around ambiguous KAIST boxes')
    parser.add_argument('--kaist-ignore-border-weight', type=float, default=0.0,
                        help='objectness loss weight for ignore-border cells; 0 keeps legacy hard ignore behavior')
    parser.add_argument('--kaist-ignore-box-pad', type=float, default=0.03,
                        help='relative padding used when rasterizing KAIST ambiguous boxes for ignore-aware loss')
    parser.add_argument('--drr-aux-weight', type=float, default=0.0,
                        help='weak box-supervised auxiliary loss weight for DASR/DRR reliability maps')
    parser.add_argument('--drr-aux-warmup', type=float, default=2.0,
                        help='epochs used to linearly ramp the DRR auxiliary loss')
    parser.add_argument('--drr-aux-fg-gain', type=float, default=1.0,
                        help='foreground-mask term gain inside the DRR auxiliary loss')
    parser.add_argument('--drr-aux-conflict-gain', type=float, default=0.35,
                        help='gain for suppressing DRR background-conflict responses inside GT boxes')
    parser.add_argument('--drr-aux-rgb-bg-gain', type=float, default=0.15,
                        help='gain for reducing daytime RGB/background-conflict reliability outside GT boxes')
    parser.add_argument('--drr-aux-day-gain', type=float, default=0.25,
                        help='weak visible-brightness scene-router supervision gain for DRR')
    parser.add_argument('--drr-aux-box-pad', type=float, default=0.08,
                        help='relative padding added to GT boxes when building DRR foreground masks')
    parser.add_argument('--drr-aux-ambiguity-gain', type=float, default=0.0,
                        help='auxiliary gain for learning countability/ambiguity maps from KAIST ignore boxes')
    parser.add_argument('--drr-aux-ignore-box-pad', type=float, default=0.04,
                        help='relative padding added to ignore boxes when building DRR ambiguity masks')
    parser.add_argument('--drr-aux-group-gain', type=float, default=0.0,
                        help='auxiliary gain for learning groupness maps from KAIST people labels')
    parser.add_argument('--drr-aux-uncertainty-gain', type=float, default=0.0,
                        help='auxiliary gain for learning uncertainty maps from KAIST person?/person?a/cyclist labels')
    parser.add_argument('--drr-aux-typed-consistency-gain', type=float, default=0.0,
                        help='auxiliary gain enforcing ambiguity to dominate typed groupness/uncertainty cues')
    parser.add_argument('--drr-aux-typed-box-pad', type=float, default=0.04,
                        help='relative padding added to typed ambiguous boxes when building semantic masks')
    parser.add_argument('--drr-aux-typed-rank-margin', type=float, default=0.15,
                        help='minimum score gap required between typed semantic positives and clean negatives')
    parser.add_argument('--drr-aux-typed-consistency-margin', type=float, default=0.05,
                        help='minimum score gap required for ambiguity to dominate typed semantic causes')
    parser.add_argument('--drr-aux-typed-residual-scale', type=float, default=0.0,
                        help='small bounded residual scale applied from typed semantic probes into the refinement path')
    parser.add_argument('--drr-aux-typed-residual-temperature', type=float, default=1.0,
                        help='temperature used when forming the typed semantic residual signal')
    parser.add_argument('--drr-aux-typed-residual-group-gain', type=float, default=1.0,
                        help='relative gain for groupness when composing the typed semantic residual signal')
    parser.add_argument('--drr-aux-typed-residual-uncertainty-gain', type=float, default=1.0,
                        help='relative gain for uncertainty when composing the typed semantic residual signal')
    parser.add_argument('--drr-aux-typed-residual-center', type=float, default=0.50,
                        help='centering value used before applying tanh to the typed semantic residual signal')
    parser.add_argument('--drr-aux-typed-countability-group-gain', type=float, default=0.0,
                        help='monotonic countability-suppression gain applied from typed groupness evidence')
    parser.add_argument('--drr-aux-typed-countability-uncertainty-gain', type=float, default=0.0,
                        help='monotonic countability-suppression gain applied from typed uncertainty evidence')
    parser.add_argument('--drr-aux-typed-countability-activation-floor', type=float, default=0.35,
                        help='typed semantic activation floor before groupness/uncertainty begin suppressing countability')
    parser.add_argument('--drr-aux-typed-countability-min-gate', type=float, default=0.70,
                        help='lower clamp for the extra typed countability gate; keeps suppression bounded')
    parser.add_argument('--kaist-typed-semantic-detach', action='store_true',
                        help='detach typed semantic probe inputs from the main detector backbone')
    parser.add_argument('--drr-aux-day-thr', type=float, default=0.28,
                        help='visible-image brightness threshold for weak DRR day-router supervision')
    parser.add_argument('--drr-aux-day-tau', type=float, default=0.08,
                        help='temperature for weak DRR day-router supervision')
    parser.add_argument('--caqh-weight', type=float, default=0.0,
                        help='Countability-Aware Quality Head loss weight for low-FPPI ranking refinement')
    parser.add_argument('--caqh-bg-weight', type=float, default=0.02,
                        help='light background supervision weight used by the Countability-Aware Quality Head')
    parser.add_argument('--caqh-ignore-gain', type=float, default=0.25,
                        help='extra supervision weight on KAIST ambiguous/ignore regions for the quality head')
    parser.add_argument('--countability-loss-weight', type=float, default=None,
                        help='RN-CAQH auxiliary countability loss weight; default None preserves legacy caqh_weight behavior')
    parser.add_argument('--countability-target-mode', type=str, default=None,
                        choices=[None, 'none', 'person_quality', 'soft_person_vs_ignore'],
                        help='RN-CAQH target mode; None keeps legacy behavior when caqh_weight > 0')
    parser.add_argument('--countability-ignore-target', type=float, default=0.35,
                        help='soft q target for ignored/ambiguous cells when using soft_person_vs_ignore')
    parser.add_argument('--pcsf-core-weight', type=float, default=0.0,
                        help='training-only protocol-conditioned soft upper-bound loss on ignore-core objectness')
    parser.add_argument('--pcsf-core-far-target', type=float, default=0.15,
                        help='upper bound for ignore-core objectness far from countable positives')
    parser.add_argument('--pcsf-core-near-target', type=float, default=0.45,
                        help='softer upper bound for ignore-core objectness near countable positives')
    parser.add_argument('--pcsf-core-near-radius', type=float, default=2.0,
                        help='grid-cell radius used to exempt ignore-core cells near matched positives')
    parser.add_argument('--rn-caqh-alpha', type=float, default=None,
                        help='optional runtime alpha for residual countability calibration')
    parser.add_argument('--rn-caqh-factor-min', type=float, default=None,
                        help='optional runtime lower clamp for residual countability factor')
    parser.add_argument('--rn-caqh-factor-max', type=float, default=None,
                        help='optional runtime upper clamp for residual countability factor')
    parser.add_argument('--rn-caqh-apply-in-inference', action='store_true',
                        help='enable RN-CAQH residual score calibration during validation/inference')
    parser.add_argument('--rcrcc-weight', type=float, default=0.0,
                        help='Reliability-Conditioned Residual Countability Calibration auxiliary loss weight')
    parser.add_argument('--rcrcc-ambiguity-gain', type=float, default=0.50,
                        help='relative gain for ambiguity when building daytime semantic risk')
    parser.add_argument('--rcrcc-conflict-gain', type=float, default=0.30,
                        help='relative gain for background conflict when building daytime semantic risk')
    parser.add_argument('--rcrcc-group-gain', type=float, default=0.70,
                        help='relative gain for groupness when building daytime semantic risk')
    parser.add_argument('--rcrcc-uncertainty-gain', type=float, default=0.15,
                        help='relative gain for uncertainty when building daytime semantic risk')
    parser.add_argument('--rcrcc-thermal-protect', type=float, default=0.35,
                        help='thermal protection factor that reduces daytime residual suppression in night-stable regions')
    parser.add_argument('--rcrcc-risk-threshold', type=float, default=0.12,
                        help='minimum local risk required before local positive-vs-ignore ranking is enforced')
    parser.add_argument('--rcrcc-rank-margin', type=float, default=0.10,
                        help='target local positive-minus-ignore margin for the residual countability head')
    parser.add_argument('--rcrcc-far-target', type=float, default=0.22,
                        help='soft upper bound for risky ignore regions far from matched positives')
    parser.add_argument('--rcrcc-near-target', type=float, default=0.40,
                        help='soft upper bound for risky ignore regions near matched positives')
    parser.add_argument('--rcrcc-positive-target', type=float, default=0.78,
                        help='minimum desired residual countability confidence over matched positives')
    parser.add_argument('--rcrcc-positive-thermal-boost', type=float, default=0.12,
                        help='extra positive target boost in thermal-stable regions')
    parser.add_argument('--rcrcc-positive-keep-weight', type=float, default=0.75,
                        help='weight for preserving positive countability confidence')
    parser.add_argument('--rcrcc-rank-weight', type=float, default=0.50,
                        help='weight for local positive-vs-ignore margin preservation')
    parser.add_argument('--rcrcc-halo-kernel', type=int, default=5,
                        help='pooling kernel used to define local ignore neighborhoods around true positives')
    parser.add_argument('--pcsf-factorized-weight', type=float, default=0.0,
                        help='master auxiliary loss weight for the explicit Round 2H H/C/G/U factorized head')
    parser.add_argument('--pcsf-h-weight', type=float, default=1.0,
                        help='humanness supervision weight inside the Round 2H factorized objective')
    parser.add_argument('--pcsf-c-weight', type=float, default=1.0,
                        help='countability supervision weight inside the Round 2H factorized objective')
    parser.add_argument('--pcsf-g-weight', type=float, default=0.35,
                        help='groupness alignment weight inside the Round 2H factorized objective')
    parser.add_argument('--pcsf-u-weight', type=float, default=0.20,
                        help='uncertainty alignment weight inside the Round 2H factorized objective')
    parser.add_argument('--pcsf-bias-weight', type=float, default=0.35,
                        help='local protocol-bias supervision weight inside the Round 2H factorized objective')
    parser.add_argument('--pcsf-pair-rank-weight', type=float, default=0.70,
                        help='instance-level positive-vs-ignore final-score ranking weight for Round 2H')
    parser.add_argument('--pcsf-semantic-rank-weight', type=float, default=0.35,
                        help='instance-level semantic-score ranking weight for Round 2H')
    parser.add_argument('--pcsf-pair-rank-margin', type=float, default=0.10,
                        help='target positive-minus-ignore ranking margin for Round 2H pairwise supervision')
    parser.add_argument('--pcsf-pair-radius', type=float, default=2.0,
                        help='local grid-cell radius used to mine hard ignore negatives around each positive anchor')
    parser.add_argument('--pcsf-ignore-target', type=float, default=0.25,
                        help='soft countability target assigned to ignore-like human regions in Round 2H')
    parser.add_argument('--pcsf-background-weight', type=float, default=0.03,
                        help='small background supervision weight for Round 2H humanness/countability maps')
    parser.add_argument('--pcsf-group-floor', type=float, default=0.50,
                        help='activation floor used when converting auxiliary groupness into a Round 2H soft target')
    parser.add_argument('--pcsf-uncertainty-floor', type=float, default=0.55,
                        help='activation floor used when converting auxiliary uncertainty into a Round 2H soft target')
    parser.add_argument('--pcsf-alpha-order-weight', type=float, default=0.10,
                        help='regularization weight encouraging alpha_G to stay stronger than alpha_U')
    parser.add_argument('--pcsf-alpha-order-margin', type=float, default=0.05,
                        help='margin used when enforcing alpha_G > alpha_U inside Round 2H')
    parser.add_argument('--pcsf-teacher-weights', type=str, default='',
                        help='optional teacher checkpoint for Round 2H positive-preserving distillation')
    parser.add_argument('--pcsf-teacher-distill-weight', type=float, default=0.0,
                        help='positive-preserving distillation weight against the frozen baseline teacher')
    parser.add_argument('--pcsf-teacher-keep-margin', type=float, default=0.02,
                        help='minimum tolerated drop below teacher positive score before a hinge penalty is applied')
    parser.add_argument('--pcsf-teacher-night-boost', type=float, default=0.30,
                        help='extra distillation emphasis on thermal-stable/night-like positives')
    parser.add_argument('--pcsf-teacher-listwise-weight', type=float, default=0.0,
                        help='listwise positive-order distillation weight against the frozen baseline teacher')
    parser.add_argument('--pcsf-teacher-listwise-temperature', type=float, default=1.0,
                        help='temperature used for listwise positive-order distillation')
    parser.add_argument('--pcsf-bias-day-gain', type=float, default=0.45,
                        help='gain on day/RGB context when building the local protocol-bias supervision target')
    parser.add_argument('--pcsf-bias-conflict-gain', type=float, default=0.35,
                        help='gain on background-conflict evidence when building the local protocol-bias target')
    parser.add_argument('--pcsf-bias-group-gain', type=float, default=0.35,
                        help='gain on groupness evidence when building the local protocol-bias target')
    parser.add_argument('--pcsf-bias-density-gain', type=float, default=0.20,
                        help='gain on local density when building the local protocol-bias target')
    parser.add_argument('--pcsf-bias-thermal-protect', type=float, default=0.35,
                        help='thermal-stability protection term subtracted from the local protocol-bias target')
    parser.add_argument('--pcsf-bias-ambiguity-gain', type=float, default=0.25,
                        help='gain on ambiguity when building the local protocol-bias target')
    parser.add_argument('--pcsf-amb-upper-weight', type=float, default=0.0,
                        help='upper-bound loss weight that only suppresses over-high countability on ignore-like regions')
    parser.add_argument('--pcsf-amb-far-target', type=float, default=0.25,
                        help='countability upper bound for ignore-like regions far from matched positives')
    parser.add_argument('--pcsf-amb-near-target', type=float, default=0.40,
                        help='softer countability upper bound for ignore-like regions near matched positives')
    parser.add_argument('--pcsf-amb-near-radius', type=float, default=2.0,
                        help='grid-cell radius used to define near-positive ignore-like regions for the upper-bound loss')
    parser.add_argument('--pcsf-protocol-weight-floor', type=float, default=0.25,
                        help='minimum retained supervision weight for protocol-invalid raw person positives inside Round 2I')
    parser.add_argument('--pcsf-protocol-rank-min-validity', type=float, default=0.15,
                        help='minimum protocol validity required before a matched positive participates in pairwise ranking')
    parser.add_argument('--pcsf-protocol-distill-min-validity', type=float, default=0.20,
                        help='minimum protocol validity required before teacher-preserving distillation is enforced')
    parser.add_argument('--pcsf-protocol-invalid-gain', type=float, default=0.35,
                        help='gain used when injecting protocol-invalid raw person mass into the Round 2I bias/risk context')
    parser.add_argument('--pcsf-duplicate-halo-weight', type=float, default=0.0,
                        help='conservative upper-bound penalty weight for near-positive duplicate halo anchors')
    parser.add_argument('--pcsf-duplicate-halo-radius', type=float, default=0.0,
                        help='grid radius used to define the duplicate-suppression halo around matched positives')
    parser.add_argument('--pcsf-duplicate-halo-target', type=float, default=0.22,
                        help='soft upper bound retained for non-matched duplicate-halo anchors')
    parser.add_argument('--pcsf-night-isolated-weight', type=float, default=0.0,
                        help='conservative upper-bound penalty weight for isolated night-time pseudo-person anchors')
    parser.add_argument('--pcsf-night-isolated-upper', type=float, default=0.18,
                        help='soft upper bound retained for isolated night-time pseudo-person anchors')
    parser.add_argument('--pcsf-night-isolated-radius', type=float, default=0.0,
                        help='positive-protection radius used before applying the isolated night-time suppression')
    parser.add_argument('--pcsf-night-isolated-min-risk', type=float, default=0.12,
                        help='minimum risk weight required before an anchor participates in isolated night-time suppression')
    parser.add_argument('--pcsf-night-isolated-conflict-gain', type=float, default=0.50,
                        help='background-conflict gain inside the isolated night-time suppression risk')
    parser.add_argument('--pcsf-night-isolated-uncertainty-gain', type=float, default=0.30,
                        help='uncertainty gain inside the isolated night-time suppression risk')
    parser.add_argument('--pcsf-night-isolated-ambiguity-gain', type=float, default=0.20,
                        help='ambiguity gain inside the isolated night-time suppression risk')
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
                        help='enable the Round 2H semantic residual calibration during validation/inference')
    parser.add_argument('--allow-partial-load', action='store_true',
                        help='load only matching checkpoint tensors; intended for default-off research heads')
    opt = parser.parse_args()

    assert not (opt.mono_rgb and opt.mono_thermal)
    assert not ((opt.mono_rgb or opt.mono_thermal) and opt.early_fusion_6ch)
    opt.mono = opt.mono_rgb or opt.mono_thermal
    opt.single_stream = opt.mono or opt.early_fusion_6ch
    #opt.rect = False

    # FQY  Flag for visualizing the paired training imgs
    global_var._init()
    global_var.set_value('flag_visual_training_dataset', False)

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    if opt.global_rank in [-1, 0]:
        # check_git_status()
        check_requirements()

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = \
            '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        log_file = Path(opt.save_dir) / 'console.log'
        set_logging(opt.global_rank, file=log_file, mode='a')
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve))
        save_dir = Path(opt.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        log_file = save_dir / 'console.log'
        set_logging(opt.global_rank, file=log_file, mode='w')

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard

            train_rgb_ir(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
