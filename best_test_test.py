import os
import random
import dill
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from trajectory_datasets import TrajectoryDataset, trajectory_collate
from base_models_test import LinearPredictor
from losses import trajectory_loss


# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# util
# =========================
def get_lambda_cls(epoch):
    if epoch < 20:
        return 0.0
    elif epoch < 60:
        return 0.4 * (epoch - 30) / 30
    else:
        return 0.4


# =========================
# metrics
# =========================
def min_ade(pred, gt):
    """
    pred: (K, B, T, 2)
    gt:   (B, T, 2)
    Returns: (B,)
    """
    K, B, T, _ = pred.shape
    gt = gt.unsqueeze(0).expand(K, -1, -1, -1)
    ade = torch.norm(pred - gt, dim=-1).mean(dim=-1)  # (K, B)
    return ade.min(dim=0)[0]


def min_fde(pred, gt):
    """
    pred: (K, B, T, 2)
    gt:   (B, T, 2)
    Returns: (B,)
    """
    K, B, T, _ = pred.shape
    gt_last = gt[:, -1].unsqueeze(0).expand(K, -1, -1)
    fde = torch.norm(pred[:, :, -1] - gt_last, dim=-1)
    return fde.min(dim=0)[0]


# =========================
# evaluation
# =========================
@torch.no_grad()
def evaluate_test(model, loader, device):
    """
    Test-time evaluation:
    only minADE / minFDE, no mode selection
    """
    model.eval()
    total_ade, total_fde = 0.0, 0.0
    total_samples = 0

    for x, y, vel, pos, nei_lists, batch_splits in loader:
        x = x.to(device)
        y = y.to(device)
        vel = vel.to(device)
        pos = pos.to(device)
        B = x.size(0)
        
        pred, _, _ = model(x, vel, pos, nei_lists, batch_splits)

        total_ade += min_ade(pred, y).sum().item()
        total_fde += min_fde(pred, y).sum().item()
        total_samples += B

    return total_ade / total_samples, total_fde / total_samples


# =========================
# train
# =========================
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # -------- load env --------
    train_env = dill.load(open(args.train_pkl, "rb"))
    val_env = dill.load(open(args.val_pkl, "rb"))
    test_env  = dill.load(open(args.test_pkl, "rb"))

    train_set = TrajectoryDataset(
        train_env, args.obs_length, args.pred_length, args.attention_radius
    )
    val_set = TrajectoryDataset(
        val_env, args.obs_length, args.pred_length, args.attention_radius
    )
    test_set = TrajectoryDataset(
        test_env, args.obs_length, args.pred_length, args.attention_radius
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=trajectory_collate,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=trajectory_collate,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=trajectory_collate,
        num_workers=0,
        pin_memory=True,
    )

    # -------- model --------
    model = LinearPredictor(args, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_ade = float('inf')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # =========================
    # training loop
    # =========================
    print("[*] Starting Training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+20}/{args.epochs}", leave=False)
        for x, y, vel, pos, nei_lists, batch_splits in train_loader:
            x = x.to(device)
            y = y.to(device)
            vel = vel.to(device)
            pos = pos.to(device)

            pred, mode_logits, _ = model(
                x, vel, pos, nei_lists, batch_splits
            )

            lambda_cls = get_lambda_cls(epoch)
            lambda_div = 0.0
            lambda_ent = 0.0

            loss = trajectory_loss(
                pred,
                y,
                mode_logits,
                lambda_div=lambda_div,
                lambda_cls=lambda_cls,
                lambda_ent=lambda_ent,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        # ===== evaluation =====
        val_minade, val_minfde = evaluate_test(
            model, val_loader, device
        )
        # test_minade, test_minfde = evaluate_test(
        #     model, test_loader, device
        # )


        if val_minade < best_val_ade:
            best_val_ade = val_minade
            save_path = os.path.join(args.ckpt_dir, "eth_best_on_val.pth")
            torch.save(model.state_dict(), save_path)
            best_flag = "★"
        else:
            best_flag = ""

        print(f"[Epoch {epoch+1:03d}] Train Loss: {avg_train_loss:.4f} | "
              f"Val minADE: {val_minade:.4f} | Val minFDE: {val_minfde:.4f} {best_flag}")


# =========================
# entry
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--train_pkl",
        type=str,
        default="processed_data_noise/eth_train.pkl",
    )
    parser.add_argument(
        "--val_pkl",
        type=str,
        default="processed_data_noise/eth_val.pkl",
    )
    parser.add_argument(
        "--test_pkl",
        type=str,
        default="processed_data_noise/eth_test.pkl",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints",
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=0, 
        help="Dataloader workers (set >0 if not on Windows)",
        )

    # Environment & Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs_length", type=int, default=8)
    parser.add_argument("--pred_length", type=int, default=12)
    parser.add_argument("--attention_radius", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Model args
    parser.add_argument("--input_size", type=int, default=4)
    parser.add_argument("--output_size", type=int, default=2)
    parser.add_argument("--x_encoder_head", type=int, default=4)
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--social_ctx_dim", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=20)

    args = parser.parse_args()
    main(args)
