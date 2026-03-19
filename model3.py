import os
import copy
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm




DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True

EPOCHS = 60
MIN_EPOCHS = 20
BATCH_SIZE = 24        # Spatial transformer uses more memory
LR = 7e-9
WEIGHT_DECAY = 1e-6
PATIENCE = 10
LAMBDA_ROT = 0.3

print("Using Device:", DEVICE)



class CubeSatPoseDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row.IMG_NUM)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        pose = torch.tensor(
            [row.X, row.Y, row.Z, row.Q1, row.Q2, row.Q3, row.W],
            dtype=torch.float32
        )

        return image, pose




class PoseModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.convnext_small(
            weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        )

        # Remove classifier
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.feature_dim = 768  # Final channel size of convnext_small

        # Lightweight Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.feature_dim,
                nhead=2,
                dim_feedforward=512,
                dropout=0.2,
                batch_first=True
            ),
            num_layers=1
        )

        self.dropout = nn.Dropout(0.3)
        self.regressor = nn.Linear(self.feature_dim, 7)

    def forward(self, x):

        # Extract spatial features
        feats = self.backbone.features(x)   # (B, C, H, W)

        B, C, H, W = feats.shape

        # Convert to tokens
        tokens = feats.view(B, C, -1)       # (B, C, H*W)
        tokens = tokens.permute(0, 2, 1)    # (B, H*W, C)

        # Transformer
        tokens = self.transformer(tokens)

        # Global pooling
        pooled = tokens.mean(dim=1)
        pooled = self.dropout(pooled)

        return self.regressor(pooled)



def normalize_quaternion(q):
    return q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)


def pose_loss(pred, gt):

    pos_loss = torch.mean((pred[:, :3] - gt[:, :3]) ** 2)

    q_pred = normalize_quaternion(pred[:, 3:])
    q_gt = normalize_quaternion(gt[:, 3:])

    dot = torch.abs(torch.sum(q_pred * q_gt, dim=1))
    rot_loss = 1.0 - dot.mean()

    return pos_loss + LAMBDA_ROT * rot_loss




def train_one_fold(model, train_loader, val_loader):

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    scaler = torch.cuda.amp.GradScaler()

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0

    for epoch in range(EPOCHS):

        # ================= TRAIN =================
        model.train()
        train_loss = 0

        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch+1}/{EPOCHS} [Train]",
                         leave=False)

        for x, y in train_bar:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = model(x)
                loss = pose_loss(out, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                with torch.cuda.amp.autocast():
                    out = model(x)
                    val_loss += pose_loss(out, y).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        tqdm.write(
            f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}"
        )

        # Early stopping (after minimum epochs)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch + 1 >= MIN_EPOCHS and wait >= PATIENCE:
            tqdm.write("Early stopping triggered.")
            break

    return best_state, best_val



def main():

    CSV_FILE = "pose_cleaned_data_with_quat.csv"
    IMG_DIR = "images"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomRotation(3),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    data = pd.read_csv(CSV_FILE)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_global_val = float("inf")
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):

        print(f"\n--- Fold {fold} ---")

        train_ds = CubeSatPoseDataset(data.iloc[train_idx], IMG_DIR, transform)
        val_ds = CubeSatPoseDataset(data.iloc[val_idx], IMG_DIR, transform)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        model = PoseModel().to(DEVICE)

        state, best_val = train_one_fold(model, train_loader, val_loader)

        print(f"Fold {fold} Best Val Loss: {best_val:.6f}")

        if best_val < best_global_val:
            best_global_val = best_val
            best_model_state = copy.deepcopy(state)

    torch.save(best_model_state, "best_convnext_small_transformer.pth")
    print("\nBest Model Saved.")


if __name__ == "__main__":
    main()