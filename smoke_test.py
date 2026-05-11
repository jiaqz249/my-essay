import torch
import numpy as np
import dill
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import DataLoader

from trajectory_datasets import TrajectoryDataset, trajectory_collate
from base_models_test import LinearPredictor
from losses import trajectory_loss



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


# ---------- 超参数（和训练保持一致） ----------
args = SimpleNamespace(
    input_size=4,
    output_size=2,
    obs_length=8,
    pred_length=12,
    x_encoder_head=4,
    embedding_size=64,
    social_ctx_dim=64,
    num_samples=20,
    batch_size=4,        # 用小 batch
    attention_radius=3.0,
    lr=1e-4,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------- 1. 加载环境并构建小数据集 ----------
print("\n[1] Loading environment...")
env = dill.load(open("processed_data_noise/eth_train.pkl", "rb"))
dataset = TrajectoryDataset(env, args.obs_length, args.pred_length, args.attention_radius)

print(f"Dataset size: {len(dataset)} windows")
if len(dataset) == 0:
    print("❌ Dataset is empty, check preprocessing!")
    exit()

# 取一个样本看看内部形状
sample = dataset[0]
print(f"  x shape: {sample['x'].shape}   (expect [N_agents, 8, 4])")
print(f"  y shape: {sample['y'].shape}   (expect [N_agents, 12, 2])")
print(f"  pos shape: {sample['pos'].shape} (expect [N_agents, 2])")
print(f"  vel shape: {sample['vel'].shape} (expect [N_agents, 2])")
print(f"  nei shape: {sample['nei'].shape} (expect [8, N_agents, N_agents])")
print("✅ Single sample shapes OK")

# ---------- 2. DataLoader ----------
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=trajectory_collate,
)

batch = next(iter(loader))
xs, ys, vels, poss, nei_lists, batch_splits = batch
print("\n[2] Batch shapes:")
print(f"  xs: {xs.shape}   (B_total, 8, 4)")
print(f"  ys: {ys.shape}   (B_total, 12, 2)")
print(f"  vels: {vels.shape} (B_total, 2)")
print(f"  poss: {poss.shape} (B_total, 2)")
print(f"  nei_lists len: {len(nei_lists)} per scene")
print(f"  batch_splits: {batch_splits}")
print("✅ DataLoader output OK")

# ---------- 3. 模型前向 ----------
print("\n[3] Model forward...")
model = LinearPredictor(args, device).to(device)
xs, ys, vels, poss = xs.to(device), ys.to(device), vels.to(device), poss.to(device)

# 注意：nei_lists 里的张量也要移到 device，collate 后它们还是 CPU 上
nei_lists = [n.to(device) for n in nei_lists]

pred, mode_logits, _ = model(xs, vels, poss, nei_lists, batch_splits)
print(f"  pred: {pred.shape}   (expect {args.num_samples}, B_total, 12, 2)")
print(f"  mode_logits: {mode_logits.shape if mode_logits is not None else None}")
# 检查是否有 NaN/Inf
print(f"  pred min/max: {pred.min().item():.4f} / {pred.max().item():.4f}")
assert not torch.isnan(pred).any(), "❌ pred contains NaN!"
print("✅ Forward pass OK")

# ---------- 4. 损失计算与反向传播 ----------
print("\n[4] Loss + backward...")
loss = trajectory_loss(
    pred, ys,
    mode_logits,
    lambda_div=0.0,
    lambda_cls=0.0,
    lambda_ent=0.0,
)
print(f"  Loss value: {loss.item():.4f}")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer.zero_grad()
loss.backward()
# 检查梯度
max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
print(f"  Max gradient: {max_grad:.6f}")
optimizer.step()
print("✅ Backward pass & step OK")

# ---------- 5. 评估函数快速验证 ----------
print("\n[5] Evaluation quick check (with progress)...")
total_ade, total_fde = 0.0, 0.0
total_samples = 0

for batch in tqdm(loader, desc="Eval", ncols=80):
    x, y, vel, pos, nei_lists, batch_splits = batch
    x = x.to(device)
    y = y.to(device)
    vel = vel.to(device)
    pos = pos.to(device)
    nei_lists = [n.to(device) for n in nei_lists]

    with torch.no_grad():
        pred, _, _ = model(x, vel, pos, nei_lists, batch_splits)
        # 评估逻辑复用
        ade = min_ade(pred, y).mean().item()
        fde = min_fde(pred, y).mean().item()

    total_ade += ade * x.size(0)
    total_fde += fde * x.size(0)
    total_samples += x.size(0)

print(f"  minADE: {total_ade / total_samples:.4f}, minFDE: {total_fde / total_samples:.4f}")
print("✅ Evaluation returns finite numbers")