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

from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


# ======================================================
# CONFIGURATION
# ======================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 32
LR = 3e-5
PATIENCE = 10
LAMBDA_ROT = 3.0

print("Using Device:", DEVICE)


# ======================================================
# DATASET
# ======================================================

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


# ======================================================
# MODEL
# ======================================================

class PoseModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        if model_name == "efficientnet_b0":
            backbone = EfficientNet.from_pretrained("efficientnet-b0")
            self.backbone = backbone.extract_features
            self.feature_dim = 1280
            self.use_transformer = True

        elif model_name == "efficientnet_b3":
            backbone = EfficientNet.from_pretrained("efficientnet-b3")
            self.backbone = backbone.extract_features
            self.feature_dim = 1536
            self.use_transformer = True

        elif model_name == "swin_tiny":
            backbone = models.swin_t(
                weights=models.Swin_T_Weights.IMAGENET1K_V1
            )
            self.feature_dim = backbone.head.in_features
            backbone.head = nn.Identity()
            self.backbone = backbone
            self.use_transformer = False

        elif model_name == "convnext_tiny":
            backbone = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            )
            self.feature_dim = backbone.classifier[2].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
            self.use_transformer = False

        else:
            raise ValueError("Unknown model")

        if self.use_transformer:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.feature_dim,
                    nhead=4,
                    dim_feedforward=512,
                    dropout=0.2,
                    batch_first=True
                ),
                num_layers=1
            )

        self.dropout = nn.Dropout(0.3)
        self.regressor = nn.Linear(self.feature_dim, 7)

    def forward(self, x):

        if self.use_transformer:
            feats = self.backbone(x)
            B, C, H, W = feats.shape
            tokens = feats.view(B, C, -1).permute(0, 2, 1)
            tokens = self.transformer(tokens)
            pooled = tokens.mean(dim=1)
            pooled = self.dropout(pooled)
            return self.regressor(pooled)

        feats = self.backbone(x)
        feats = self.dropout(feats)
        return self.regressor(feats)


# ======================================================
# LOSS
# ======================================================

def normalize_quaternion(q):
    return q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)


def pose_loss(pred, gt, lambda_rot=LAMBDA_ROT):

    pos_loss = torch.mean((pred[:, :3] - gt[:, :3]) ** 2)

    q_pred = normalize_quaternion(pred[:, 3:])
    q_gt = normalize_quaternion(gt[:, 3:])

    dot = torch.abs(torch.sum(q_pred * q_gt, dim=1))
    rot_loss = 1.0 - dot.mean()

    return pos_loss + lambda_rot * rot_loss


# ======================================================
# TRAIN ONE FOLD
# ======================================================

def train_one_fold(model, train_loader, val_loader):

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0

    for epoch in range(EPOCHS):

        # ===== TRAIN =====
        model.train()
        train_loss = 0

        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch+1}/{EPOCHS} [Train]",
                         leave=False)

        for x, y in train_bar:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = pose_loss(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        preds, gts = [], []

        val_bar = tqdm(val_loader,
                       desc=f"Epoch {epoch+1}/{EPOCHS} [Val]",
                       leave=False)

        with torch.no_grad():
            for x, y in val_bar:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)

                batch_loss = pose_loss(out, y).item()
                val_loss += batch_loss

                val_bar.set_postfix(val_loss=f"{batch_loss:.4f}")

                preds.append(out.cpu().numpy())
                gts.append(y.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        tqdm.write(
            f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                tqdm.write("Early stopping triggered.")
                break

    preds = np.vstack(preds)
    gts = np.vstack(gts)

    total_mse = mean_squared_error(gts, preds)
    trans_mse = mean_squared_error(gts[:, :3], preds[:, :3])
    rot_mse = mean_squared_error(gts[:, 3:], preds[:, 3:])

    return best_state, total_mse, trans_mse, rot_mse


# ======================================================
# MAIN
# ======================================================

def main():

    CSV_FILE = "pose_cleaned_data_with_quat.csv"
    IMG_DIR = "images"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.3, 0.3),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    data = pd.read_csv(CSV_FILE)

    models_list = [
        "swin_tiny",
        "efficientnet_b0",
        "convnext_tiny"
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    best_global_mse = float("inf")
    best_model_state = None
    best_model_name = None

    for model_name in models_list:

        print("\n==============================")
        print(f"Training Model: {model_name}")
        print("==============================")

        fold_total, fold_trans, fold_rot = [], [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):

            print(f"\n--- Fold {fold} ---")

            train_ds = CubeSatPoseDataset(data.iloc[train_idx], IMG_DIR, transform)
            val_ds = CubeSatPoseDataset(data.iloc[val_idx], IMG_DIR, transform)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

            model = PoseModel(model_name).to(DEVICE)

            state, total_mse, trans_mse, rot_mse = train_one_fold(
                model, train_loader, val_loader
            )

            print(f"Fold {fold} Results:")
            print(f"  Total MSE: {total_mse:.6f}")
            print(f"  Translation MSE: {trans_mse:.6f}")
            print(f"  Rotation MSE: {rot_mse:.6f}")

            fold_total.append(total_mse)
            fold_trans.append(trans_mse)
            fold_rot.append(rot_mse)

            if total_mse < best_global_mse:
                best_global_mse = total_mse
                best_model_state = copy.deepcopy(state)
                best_model_name = model_name

        results[model_name] = {
            "Total_MSE": np.mean(fold_total),
            "Trans_MSE": np.mean(fold_trans),
            "Rot_MSE": np.mean(fold_rot)
        }

    print("\n===== FINAL MODEL RANKING =====")

    ranked = sorted(results.items(), key=lambda x: x[1]["Total_MSE"])

    for i, (name, metrics) in enumerate(ranked, 1):
        print(f"\nRank {i}: {name}")
        print(f"  Avg Total MSE: {metrics['Total_MSE']:.6f}")
        print(f"  Avg Translation MSE: {metrics['Trans_MSE']:.6f}")
        print(f"  Avg Rotation MSE: {metrics['Rot_MSE']:.6f}")

    torch.save(best_model_state, "best_model.pth")
    print(f"\nBest Model Saved: {best_model_name}")


if __name__ == "__main__":
    main()