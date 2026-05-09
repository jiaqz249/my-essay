import os
import dill
import numpy as np
import matplotlib.pyplot as plt

# ========================================
# 1. 选择你要检查的 pkl 文件
# ========================================
file_to_check = 'zara2_val.pkl' 
file_path = os.path.join('processed_data_noise', file_to_check)

print(f"Loading data from {file_path} ...")
with open(file_path, 'rb') as f:
    env = dill.load(f)

print("\n" + "="*40)
print(" 整体环境信息 (Environment Info)")
print("="*40)
print(f"包含的场景 (Scenes) 数量: {len(env.scenes)}")

scene = env.scenes[0]
print("\n" + "="*40)
print(f" 场景信息 (Scene Info): {scene.name}")
print("="*40)
print(f"总时间步 (Timesteps): {scene.timesteps}")
print(f"时间间隔 (dt): {scene.dt} 秒")
print(f"场景内行人数量 (Nodes): {len(scene.nodes)}")

# ========================================
# 3. 抽查某个行人的数据 (使用 Node.timesteps 和 字典索引)
# ========================================
# 修复：直接使用作者写好的 timesteps 属性
longest_node = max(scene.nodes, key=lambda n: n.timesteps)

print("\n" + "="*40)
print(f" 抽查行人 (Node ID: {longest_node.id})")
print("="*40)
print(f"出现的第一帧 (First Timestep): {longest_node.first_timestep}")
print(f"轨迹总长度 (Frames): {longest_node.timesteps}")

# 【终极修复】：使用字典字典作为列索引，这会直接返回纯净的 Numpy 数组！
pos_data = longest_node.data[:, {'position': ['x', 'y']}]
vel_data = longest_node.data[:, {'velocity': ['x', 'y']}]

print("前 5 帧 position [pos_x, pos_y]:")
print(pos_data[:5]) 
print("前 5 帧 velocity [vel_x, vel_y]:")
print(vel_data[:5])

has_nan = np.isnan(pos_data).any() or np.isnan(vel_data).any()
if has_nan:
    print("\n[警告] 该节点数据中存在 NaN (缺失值)！")
else:
    print("\n[正常] 该节点数据完整，无 NaN。")


# ========================================
# 4. 可视化：把这个场景里所有人的轨迹画出来
# ========================================
def visualize_scene(scene, title=""):
    plt.figure(figsize=(10, 8))
    
    for node in scene.nodes:
        if node.type.name == 'PEDESTRIAN':
            # 【终极修复】：使用字典来提取所需特征
            pos = node.data[:, {'position': ['x', 'y']}]
            x = pos[:, 0]
            y = pos[:, 1]
            
            line, = plt.plot(x, y, linewidth=1.5, alpha=0.7)
            plt.scatter(x[0], y[0], color=line.get_color(), marker='o', s=30, zorder=3)
            plt.scatter(x[-1], y[-1], color=line.get_color(), marker='*', s=50, zorder=3)
            
    plt.title(f"Bird's-eye View: {title}")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')  
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Start Point'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=12, label='End Point')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.show()

print("\n正在生成轨迹可视化图像，请查看弹出的窗口...")
visualize_scene(scene, title=f"Trajectories in {scene.name}")