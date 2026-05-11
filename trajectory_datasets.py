import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Build training samples from processed Environment / Scene / Node objects.
    Each sample corresponds to ONE scene window (multiple agents).
    """

    def __init__(self, env, obs_len: int, pred_len: int, attention_radius: float):
        self.env = env
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.attention_radius = attention_radius

        # index: list of (scene, start_t)
        self.index = []
        self._build_index()

    def _build_index(self):
        """
        Enumerate all valid (scene, t_start) windows.
        """
        for scene in self.env.scenes:
            T = int(scene.timesteps)
            for t in range(0, T - self.obs_len - self.pred_len + 1):
                # check at least one valid agent
                valid_nodes = []
                for node in scene.nodes:
                    t0 = node.first_timestep
                    t1 = t0 + node.timesteps
                    if t >= t0 and (t + self.obs_len + self.pred_len) <= t1:
                        valid_nodes.append(node)
                if len(valid_nodes) > 0:
                    self.index.append((scene, t))

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        scene, t_start = self.index[idx]

        std_dict = self.env.standardization['PEDESTRIAN']
        px_m, px_s = std_dict['position']['x']['mean'], std_dict['position']['x']['std']
        py_m, py_s = std_dict['position']['y']['mean'], std_dict['position']['y']['std']
        vx_m, vx_s = std_dict['velocity']['x']['mean'], std_dict['velocity']['x']['std']
        vy_m, vy_s = std_dict['velocity']['y']['mean'], std_dict['velocity']['y']['std']

        xs, ys, vels, raw_poss_list = [], [], [], []

        for node in scene.nodes:
            t0 = int(node.first_timestep)
            T = int(node.timesteps)
            t1 = t0 + T

            if t_start < t0 or (t_start + self.obs_len + self.pred_len) > t1:
                continue

            s = int(t_start - t0)
            e_obs = s + self.obs_len
            e_fut = e_obs + self.pred_len

            pos_data = node.data[:, {'position': ['x', 'y']}]
            vel_data = node.data[:, {'velocity': ['x', 'y']}]

            obs_pos = pos_data[s : e_obs]
            obs_vel = vel_data[s : e_obs]
            fut_pos = pos_data[e_obs : e_fut]
            
            
            norm_obs_px = (obs_pos[:, 0] - px_m) / px_s
            norm_obs_py = (obs_pos[:, 1] - py_m) / py_s
            norm_obs_vx = (obs_vel[:, 0] - vx_m) / vx_s
            norm_obs_vy = (obs_vel[:, 1] - vy_m) / vy_s

            norm_fut_px = (fut_pos[:, 0] - px_m) / px_s
            norm_fut_py = (fut_pos[:, 1] - py_m) / py_s


            x = np.stack([norm_obs_px, norm_obs_py, norm_obs_vx, norm_obs_vy], axis=-1).astype(np.float32)
            y = np.stack([norm_fut_px, norm_fut_py], axis=-1).astype(np.float32)
            v = np.stack([norm_obs_vx, norm_obs_vy], axis=-1).astype(np.float32)
            
            xs.append(x)
            ys.append(y)
            vels.append(v[-1])
            
            raw_poss_list.append(obs_pos[-1, :2].astype(np.float32))

        xs = np.stack(xs)                 # (N, obs_len, 4)
        ys = np.stack(ys)                 # (N, pred_len, 2)
        vels = np.stack(vels)             # (N, 2)
        raw_poss_list = np.stack(raw_poss_list) # (N, 2)

        nei = self._build_neighbors(raw_poss_list)

        return {
            'x': torch.from_numpy(xs),
            'y': torch.from_numpy(ys),
            'vel': torch.from_numpy(vels),
            'pos': torch.from_numpy(raw_poss_list),
            'nei': torch.from_numpy(nei),
        }


    def _build_neighbors(self, pos):
        """
        Build adjacency matrix based on distance threshold.
        pos: (N, 2)
        """
        N = pos.shape[0]
        nei = np.zeros((self.obs_len, N, N), dtype=np.int64)
        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        mask = (dist < self.attention_radius) & (dist > 0)
        nei[:] = mask.astype(np.int64)
        return nei

def trajectory_collate(batch):
    """
    Collate multiple scenes into one batch.
    """
    xs, ys, vels, poss = [], [], [], []
    nei_lists = []
    batch_splits = []

    offset = 0
    for item in batch:
        n = item['x'].shape[0]
        xs.append(item['x'])
        ys.append(item['y'])
        vels.append(item['vel'])
        poss.append(item['pos'])
        nei_lists.append(item['nei'])
        batch_splits.append([offset, offset + n])
        offset += n

    xs = torch.cat(xs, dim=0)      # (B, H, 4)
    ys = torch.cat(ys, dim=0)      # (B, T, 2)
    vels = torch.cat(vels, dim=0)  # (B, 2)
    poss = torch.cat(poss, dim=0)  # (B, 2)

    return xs, ys, vels, poss, nei_lists, batch_splits