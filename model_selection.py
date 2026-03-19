import os
import copy
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

VARIANCE = 1.000073
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CubeSatPoseDataset(Dataset):
    def __init__(self, dataframe, images_path, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_path, row.IMG_NUM)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        pose = torch.tensor([row.X, row.Y, row.Z,
                             row.Q1, row.Q2, row.Q3, row.W],
                            dtype=torch.float32)
        return img, pose


class CNNTransformerModel(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()

        if backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone

        elif backbone_name == "resnet101":
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            self.feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone

        elif backbone_name == "densenet121":
            backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.feature_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone

        elif backbone_name == "densenet201":
            backbone = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
            self.feature_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone

        elif backbone_name == "mobilenet_v3":
            backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            self.feature_dim = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone

        elif backbone_name == "convnext":
            backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.feature_dim = backbone.classifier[2].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone

        elif backbone_name == "efficientnet_b0":
            backbone = EfficientNet.from_pretrained('efficientnet-b0')
            self.feature_dim = backbone._fc.in_features
            backbone._fc = nn.Identity()
            self.backbone = backbone

        elif backbone_name == "efficientnet_b3":
            backbone = EfficientNet.from_pretrained('efficientnet-b3')
            self.feature_dim = backbone._fc.in_features
            backbone._fc = nn.Identity()
            self.backbone = backbone

        elif backbone_name == "swin":
            backbone = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            self.feature_dim = backbone.head.in_features
            backbone.head = nn.Identity()
            self.backbone = backbone

        else:
            raise ValueError("Unknown backbone")

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.feature_dim,
                nhead=8,
                dim_feedforward=1024,
                batch_first=True),
            num_layers=2)

        self.regressor = nn.Linear(self.feature_dim, 7)

    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = torch.flatten(features, 1)
        seq = features.unsqueeze(1)
        transformed = self.transformer(seq).squeeze(1)
        return self.regressor(transformed)


def normalize_quaternion(q):
    return q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)


def pose_loss(y_pred, y_true):
    pos_mse = torch.mean((y_pred[:, :3] - y_true[:, :3]) ** 2)
    pred_q = normalize_quaternion(y_pred[:, 3:])
    true_q = normalize_quaternion(y_true[:, 3:])
    quat_mse = torch.mean((pred_q - true_q) ** 2)
    return pos_mse + quat_mse


# ================= MODIFIED TRAIN FUNCTION =================

def train_model(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    val_losses = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epoch = 0
    max_epochs = 70

    while epoch < max_epochs:
        epoch += 1
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = pose_loss(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += pose_loss(model(x), y).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss:.6f}")

        val_losses.append(val_loss)

        # Update best weights
        if val_loss == min(val_losses):
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        # Early stopping only after epoch 27
        if epoch > 27:
            if len(val_losses) > 2 and val_losses[-1] > val_losses[-3]:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, best_epoch


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    acc = 1 - (rmse / VARIANCE)
    return mse, rmse, acc


def main():
    CSV_FILE = "C:/Users/Admin/Desktop/pose_estimation/synthetic_cubesat/dataset/pose_cleaned_data_with_quat.csv"
    IMAGES_PATH = "C:/Users/Admin/Desktop/pose_estimation/synthetic_cubesat/dataset/images"

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    data = pd.read_csv(CSV_FILE)

    backbones = [
        "resnet50","resnet101","densenet121","densenet201",
        "mobilenet_v3","convnext","efficientnet_b0",
        "efficientnet_b3","swin","resnet50"
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_overall_acc = -float("inf")
    best_model_state = None
    best_model_name = None

    for backbone in backbones:
        print(f"\nTraining Model: {backbone}+Transformer")
        mse_list, rmse_list, epoch_list = [], [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            print(f"\nFold {fold+1}")

            train_df = data.iloc[train_idx]
            val_df = data.iloc[val_idx]

            train_ds = CubeSatPoseDataset(train_df, IMAGES_PATH, transform)
            val_ds = CubeSatPoseDataset(val_df, IMAGES_PATH, transform)

            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=16)

            model = CNNTransformerModel(backbone).to(DEVICE)
            model, best_epoch = train_model(model, train_loader, val_loader)

            preds, gts = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(DEVICE)
                    preds.append(model(x).cpu().numpy())
                    gts.append(y.numpy())

            mse, rmse, acc = compute_metrics(np.vstack(gts), np.vstack(preds))
            mse_list.append(mse)
            rmse_list.append(rmse)
            epoch_list.append(best_epoch)

        avg_acc = 1 - (np.mean(rmse_list)/VARIANCE)

        results[backbone+"+Transformer"] = {
            "MSE": np.mean(mse_list),
            "RMSE": np.mean(rmse_list),
            "Accuracy": avg_acc,
            "Epoch": int(np.mean(epoch_list))
        }

        if avg_acc > best_overall_acc:
            best_overall_acc = avg_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_model_name = backbone+"+Transformer"

    ranked = sorted(results.items(), key=lambda x: x[1]["Accuracy"], reverse=True)

    print("\nFINAL MODEL RANKING:")
    for i,(name,metrics) in enumerate(ranked,1):
        print(f"\nRank {i}: {name}")
        print(f"MSE: {metrics['MSE']:.6f}")
        print(f"RMSE: {metrics['RMSE']:.6f}")
        print(f"Accuracy: {metrics['Accuracy']:.6f}")
        print(f"Chosen Epoch: {metrics['Epoch']}")

    torch.save(best_model_state, "best_model.pth")
    print("\nBest Model Selected:", best_model_name)
    print("Saved as best_model.pth")


if __name__ == "__main__":
    main()