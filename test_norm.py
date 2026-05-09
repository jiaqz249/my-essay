import os
import numpy as np
import pandas as pd
import pickle
from environment import derivative_of

# 跟原脚本一致的参数
dt = 0.4

# ==============================
# 统计容器：位置和速度的全部值
# ==============================
all_pos_x = []
all_pos_y = []
all_vel_x = []
all_vel_y = []

# ==============================
# 1. ETH – UCY 训练集
# ==============================
for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
    data_dir = os.path.join('raw_data', desired_source, 'train')
    if not os.path.isdir(data_dir):
        continue

    for subdir, _, files in os.walk(data_dir):
        for file in files:
            if not file.endswith('.txt'):
                continue
            path = os.path.join(subdir, file)
            print(f'Processing {path}')

            # ---- 与原脚本完全相同的预处理 ----
            data = pd.read_csv(path, sep='\t', header=None,
                               names=['frame_id', 'track_id', 'pos_x', 'pos_y'])
            data['frame_id'] = pd.to_numeric(data['frame_id']) // 10
            data['frame_id'] -= data['frame_id'].min()

            # 场景内去均值（原脚本是对整个文件的所有点取 mean）
            # data['pos_x'] -= data['pos_x'].mean()
            # data['pos_y'] -= data['pos_y'].mean()

            # 遍历每个行人
            for track_id, node_df in data.groupby('track_id'):
                if len(node_df) < 2:
                    continue
                node_df = node_df.sort_values('frame_id')
                frames = node_df['frame_id'].values
                # 检查连续（可选，原脚本有断言）
                # assert np.all(np.diff(frames) == 1)

                x = node_df['pos_x'].values.astype(np.float64)
                y = node_df['pos_y'].values.astype(np.float64)

                # 位置直接加入统计
                all_pos_x.extend(x.tolist())
                all_pos_y.extend(y.tolist())

                # 速度（跟原脚本一样用 derivative_of）
                vx = derivative_of(x, dt)
                vy = derivative_of(y, dt)
                all_vel_x.extend(vx.tolist())
                all_vel_y.extend(vy.tolist())

# ==============================
# 2. Stanford Drone 训练集
# ==============================
sdd_train_path = os.path.join('raw_data', 'stanford', 'train_trajnet.pkl')
if os.path.exists(sdd_train_path):
    print(f'Processing {sdd_train_path}')
    df = pickle.load(open(sdd_train_path, 'rb'))

    for scene_id, data in df.groupby('sceneId'):
        # ---- 与原脚本完全相同的预处理 ----
        data = data.copy()
        data['frame'] = pd.to_numeric(data['frame']) // 12
        data['frame'] -= data['frame'].min()
        data['x'] = data['x'] / 50.0
        data['y'] = data['y'] / 50.0

        # 场景内去均值（原脚本是按场景 sceneId 去均值）
        # data['x'] -= data['x'].mean()
        # data['y'] -= data['y'].mean()

        # 遍历每个行人
        for track_id, node_df in data.groupby('trackId'):
            if len(node_df) < 2:
                continue
            node_df = node_df.sort_values('frame')
            frames = node_df['frame'].values
            # assert np.all(np.diff(frames) == 1)

            x = node_df['x'].values.astype(np.float64)
            y = node_df['y'].values.astype(np.float64)

            all_pos_x.extend(x.tolist())
            all_pos_y.extend(y.tolist())

            vx = derivative_of(x, dt)
            vy = derivative_of(y, dt)
            all_vel_x.extend(vx.tolist())
            all_vel_y.extend(vy.tolist())

# ==============================
# 计算全局统计量
# ==============================
mean_px = np.mean(all_pos_x)
std_px = np.std(all_pos_x)
mean_py = np.mean(all_pos_y)
std_py = np.std(all_pos_y)

mean_vx = np.mean(all_vel_x)
std_vx = np.std(all_vel_x)
mean_vy = np.mean(all_vel_y)
std_vy = np.std(all_vel_y)

# 防止 std 为 0（极端情况）
std_px = std_px if std_px > 0 else 1.0
std_py = std_py if std_py > 0 else 1.0
std_vx = std_vx if std_vx > 0 else 1.0
std_vy = std_vy if std_vy > 0 else 1.0

# ==============================
# 输出可直接用的标准化字典
# ==============================
standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': mean_px, 'std': std_px},
            'y': {'mean': mean_py, 'std': std_py}
        },
        'velocity': {
            'x': {'mean': mean_vx, 'std': std_vx},
            'y': {'mean': mean_vy, 'std': std_vy}
        }
    }
}

print("\n===== 统计结果 =====")
print("standardization = {")
print("    'PEDESTRIAN': {")
print("        'position': {")
print(f"            'x': {{'mean': {mean_px:.6f}, 'std': {std_px:.6f}}},")
print(f"            'y': {{'mean': {mean_py:.6f}, 'std': {std_py:.6f}}}")
print("        },")
print("        'velocity': {")
print(f"            'x': {{'mean': {mean_vx:.6f}, 'std': {std_vx:.6f}}},")
print(f"            'y': {{'mean': {mean_vy:.6f}, 'std': {std_vy:.6f}}}")
print("        }")
print("    }")
print("}")