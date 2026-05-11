import dill
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

from trajectory_datasets import TrajectoryDataset, trajectory_collate
from base_models_test import LinearPredictor

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
def evaluate(model, loader, device, env):
    model.eval()

    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0
    
    std_dict = env.standardization['PEDESTRIAN']['position']
    px_m, py_m = std_dict['x']['mean'], std_dict['y']['mean']
    px_s, py_s = std_dict['x']['std'], std_dict['y']['std']
    
    mean_tensor = torch.tensor([px_m, py_m], device=device)
    std_tensor = torch.tensor([px_s, py_s], device=device)

    with torch.no_grad():
        for x, y, vel, pos, nei_lists, batch_splits in loader:
            x = x.to(device)
            y = y.to(device)
            vel = vel.to(device)
            pos = pos.to(device)
            B = x.size(0)

            pred, _, _ = model(x, vel, pos, nei_lists, batch_splits)
            
            pred_real = pred * std_tensor + mean_tensor
            y_real = y * std_tensor + mean_tensor

            ade = min_ade(pred_real, y_real) # (B,)
            fde = min_fde(pred_real, y_real) # (B,)

            total_ade += ade.sum().item()
            total_fde += fde.sum().item()
            total_samples += B

    return total_ade / total_samples, total_fde / total_samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = dill.load(open("processed_data_noise/eth_test.pkl", "rb"))

    dataset = TrajectoryDataset(
        env,
        obs_len=8,
        pred_len=12,
        attention_radius=3.0
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=trajectory_collate
    )

    args = SimpleNamespace(
        input_size=4,
        output_size=2,
        obs_length=8,
        pred_length=12,     
        x_encoder_head=4,      # Transformer encoder 用
        embedding_size=64,
        social_ctx_dim=64,
        num_samples=20,
    )

    model = LinearPredictor(args, device).to(device)
    model.load_state_dict(torch.load("checkpoints/eth_best_on_test.pth"))

    ade, fde = evaluate(model, loader, device, env)

    print("\n" + "="*40)
    print(" Evaluation Results ")
    print("="*40)
    print(f"  minADE: {ade:.4f} meters")
    print(f"  minFDE: {fde:.4f} meters")
    print("="*40)


if __name__ == "__main__":
    main()
