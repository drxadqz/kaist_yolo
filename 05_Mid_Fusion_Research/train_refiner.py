import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset_refiner import KAISTRefinerDataset
from mid_fusion_refiner import MidFusionRefiner


# ===================== Config =====================
IMAGE_ROOT = r"E:\KAIST_Dataset\images"
ANNOTATION_ROOT = r"E:\KAIST_Dataset\annotations"
SAVE_DIR = r"E:\kaist_yolo\05_Mid_Fusion_Research\weights"
SAVE_NAME = "mid_fusion_refiner.pt"

EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
PATCH_SIZE = 96
MAX_SAMPLES = 20000
NUM_WORKERS = 2
SEED = 42
# ==================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train() -> None:
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = KAISTRefinerDataset(
        image_root=IMAGE_ROOT,
        annotation_root=ANNOTATION_ROOT,
        patch_size=PATCH_SIZE,
        max_samples=MAX_SAMPLES,
        seed=SEED,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check IMAGE_ROOT/ANNOTATION_ROOT paths.")

    val_len = max(1, int(0.1 * len(dataset)))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(
        dataset, [train_len, val_len], generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MidFusionRefiner(base_ch=32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val = float("inf")
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for vis_patch, ir_patch, labels in train_loader:
            vis_patch = vis_patch.to(device, non_blocking=True)
            ir_patch = ir_patch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(vis_patch, ir_patch)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for vis_patch, ir_patch, labels in val_loader:
                vis_patch = vis_patch.to(device, non_blocking=True)
                ir_patch = ir_patch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(vis_patch, ir_patch)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits)
                pred = (probs >= 0.5).float()
                correct += (pred == labels).sum().item()
                total += labels.numel()

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        acc = correct / total if total > 0 else 0.0
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={acc:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state_dict": model.state_dict(),
                "patch_size": PATCH_SIZE,
                "base_ch": 32,
            }
            torch.save(ckpt, save_path)
            print(f"Saved best checkpoint -> {save_path}")

    print("Training done.")


if __name__ == "__main__":
    train()

