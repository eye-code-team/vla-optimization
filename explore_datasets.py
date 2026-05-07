import os
import sys
import argparse
import types
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# =====================================================================
#  PATCH: Fix lerobot 0.4.4 import chain crash on Windows/Python 3.10
# =====================================================================
import lerobot
_pkg = lerobot.__path__[0]
import importlib.util

_robots_mod = types.ModuleType('lerobot.robots')
_robots_mod.__path__ = [os.path.join(_pkg, 'robots')]
_robots_mod.__package__ = 'lerobot.robots'
_spec = importlib.util.spec_from_file_location('lerobot.robots.config', os.path.join(_pkg, 'robots', 'config.py'))
_cfg_mod = importlib.util.module_from_spec(_spec)
sys.modules['lerobot.robots.config'] = _cfg_mod
_spec.loader.exec_module(_cfg_mod)
_robots_mod.RobotConfig = _cfg_mod.RobotConfig
sys.modules['lerobot.robots'] = _robots_mod

_proc_mod = types.ModuleType('lerobot.processor')
_proc_mod.__path__ = [os.path.join(_pkg, 'processor')]
_proc_mod.__package__ = 'lerobot.processor'
_proc_mod.RobotAction = dict
_proc_mod.RobotObservation = dict
_proc_mod.PolicyAction = dict
sys.modules['lerobot.processor'] = _proc_mod

_policies_mod = types.ModuleType('lerobot.policies')
_policies_mod.__path__ = [os.path.join(_pkg, 'policies')]
_policies_mod.__package__ = 'lerobot.policies'
sys.modules['lerobot.policies'] = _policies_mod

from lerobot.datasets.lerobot_dataset import LeRobotDataset


SUPPORTED_DATASETS = [
    "bnarin/so100_tic_tac_toe_we_do_it_live", "dc2ac/so100-t5", "chmadran/so100_home_dataset",
    "baladhurgesh97/so100_final_picking_3", "bnarin/so100_tic_tac_toe_move_0_0",
    "bnarin/so100_tic_tac_toe_move_1_0", "bnarin/so100_tic_tac_toe_move_2_1",
    "bnarin/so100_tic_tac_toe_move_4_0", "zaringleb/so100_cube_6_2d",
    "andlyu/so100_indoor_0", "andlyu/so100_indoor_2", "Winster/so100_sim",
    "badwolf256/so100_twin_cam_duck", "Congying1112/so100_simplepick_with_2_cameras_from_top",
    "andlyu/so100_indoor_4", "Zak-Y/so100_grap_dataset",
    "kantine/domotic_pouringCoffee_expert", "kantine/domotic_pouringCoffee_anomaly",
    "lucasngoo/so100_strawberry_grape", "kantine/domotic_makingCoffee_expert",
    "kantine/domotic_makingCoffee_anomaly", "ZGGZZG/so100_drop1",
    "kantine/industrial_soldering_expert", "kantine/industrial_soldering_anomaly",
    "Yotofu/so100_sweeper_shoes", "kantine/domotic_dishTidyUp_expert",
    "kantine/domotic_dishTidyUp_anomaly", "kantine/domotic_groceriesSorting_expert",
    "kantine/domotic_groceriesSorting_anomaly", "badwolf256/so100_twin_cam_duck_v2",
    "kantine/domotic_vegetagblesAndFruitsSorting_expert",
    "kantine/domotic_vegetagblesAndFruitsSorting_anomaly", "kantine/domotic_setTheTable_expert",
    "kantine/domotic_setTheTable_anomaly", "therarelab/so100_pick_place", "abhisb/so100_51_ep",
    "andlyu/so100_indoor_val_0", "lizi178119985/so100_jia", "badwolf256/so100_twin_cam_duck_v3",
    "andrewcole712/so100_tape_bin_place", "Gano007/so100_lolo", "Zak-Y/so100_three_cameras_dataset",
    "Gano007/so100_doliprane", "XXRRSSRR/so100_v3_num_episodes_50", "zijian2022/assemblyarm2",
    "ganker5/so100_action_20250403", "andlyu/so100_indoor_val2", "Gano007/so100_gano"
]


def _run_lerobot_dataset(dataset, out_dir, repo_id, num_episodes):
    sample = dataset[0]
    all_keys = list(sample.keys())
    image_keys = [k for k in all_keys if 'image' in k.lower() and isinstance(sample[k], torch.Tensor)]
    state_key = next((k for k in all_keys if 'state' in k.lower() and isinstance(sample[k], torch.Tensor)), None)
    action_key = 'action'
    
    print(f"\n[Thông tin chuẩn - {repo_id}]")
    print(f" - Tổng số Episodes: {dataset.num_episodes}")
    print(f" - Tổng số Frames: {len(dataset)}")
    fps = getattr(dataset, 'fps', 30)
    if not fps: fps = 30
    print(f" - FPS: {fps}")
    print(f" - Các Camera/View: {image_keys}")
    
    ep_col = dataset.hf_dataset['episode_index']
    episode_indices = {}
    for idx, ep in enumerate(ep_col):
        ep_int = ep.item() if isinstance(ep, torch.Tensor) else int(ep)
        if ep_int not in episode_indices:
            episode_indices[ep_int] = []
        episode_indices[ep_int].append(idx)
        
    all_eps = sorted(episode_indices.keys())
    target_eps = all_eps[:min(num_episodes, len(all_eps))]
    
    if len(image_keys) == 0:
        print("Dataset này không chứa dữ liệu hình ảnh (images)!")
        return

    primary_cam = image_keys[0]

    for ep_idx in target_eps:
        indices = episode_indices[ep_idx]
        n_frames = len(indices)
        vpath = out_dir / f'explore_ep_{ep_idx}.mp4'
        writer = None
        
        print(f"  -> Đang xử lý Episode {ep_idx} ({n_frames} frames)...")
        for t in tqdm(range(n_frames), leave=False, desc=f"Ep {ep_idx}"):
            frame_data = dataset[indices[t]]
            img_tensor = frame_data[primary_cam].numpy()
            
            if img_tensor.shape[0] <= 4:
                img = np.transpose(img_tensor, (1, 2, 0))
            else:
                img = img_tensor

            if img.max() <= 1.0: 
                img = (img * 255).clip(0, 255).astype(np.uint8)
            else: 
                img = img.clip(0, 255).astype(np.uint8)
                
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            if writer is None:
                fh, fw = img_bgr.shape[:2]
                if fw < 400:
                    fw, fh = fw * 2, fh * 2
                writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))
            
            img_bgr = cv2.resize(img_bgr, (fw, fh))
            action_val = frame_data[action_key].numpy()
            cv2.putText(img_bgr, f'Act[0]: {action_val[0]:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(img_bgr, f'Ep: {ep_idx} | Frame: {t}/{n_frames}', (10, fh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            writer.write(img_bgr)
            
        if writer is not None:
            writer.release()
            
    print(f"\nHoàn tất! Video khảo sát lưu tại: {out_dir}")

def _run_raw_hf_dataset(repo_id, out_dir, num_episodes):
    from datasets import load_dataset
    print("  Đang fetch raw dataset qua 'datasets'...")
    hf_ds = load_dataset(repo_id, split="train")
    
    all_keys = list(hf_ds.features.keys())
    image_keys = [k for k in all_keys if 'image' in k.lower() or 'cam' in k.lower() or 'video' in k.lower()]
    action_key = 'action' if 'action' in all_keys else None
    
    print(f"\n[Thông tin Raw HuggingFace - {repo_id}]")
    print(f" - Tổng số Frames: {len(hf_ds)}")
    print(f" - Các Camera/View: {image_keys}")
    
    # Tìm cột episode_index
    ep_key = 'episode_index' if 'episode_index' in all_keys else None
    if ep_key is None:
        print("Không tìm thấy cột phân chia Episode. Trích xuất mặc định 100 frame đầu tiên thành 1 clip.")
        ep_col = [0] * len(hf_ds)
    else:
        ep_col = hf_ds[ep_key]
        
    episode_indices = {}
    for idx, ep_val in enumerate(ep_col):
        ep_int = int(ep_val)
        if ep_int not in episode_indices:
            episode_indices[ep_int] = []
        episode_indices[ep_int].append(idx)
        
    all_eps = sorted(episode_indices.keys())
    target_eps = all_eps[:min(num_episodes, len(all_eps))]
    
    if len(image_keys) == 0:
        print(f"Dataset này không chứa cột hình ảnh chuẩn (không có chữ image/cam/video)!\nDanh sách tất cả các cột đang có: {all_keys}")
        return
        
    primary_cam = image_keys[0]

    for ep_idx in target_eps:
        indices = episode_indices[ep_idx]
        n_frames = len(indices)
        vpath = out_dir / f'explore_ep_{ep_idx}.mp4'
        writer = None
        
        print(f"  -> Đang xử lý Episode {ep_idx} ({n_frames} frames)...")
        for t in tqdm(range(n_frames), leave=False, desc=f"Ep {ep_idx}"):
            frame_data = hf_ds[indices[t]]
            
            # RAW HF returns PIL Images for image features
            img_obj = frame_data[primary_cam]
            if hasattr(img_obj, 'convert'): # PIL Image
                img_rgb = np.array(img_obj.convert("RGB"))
            else:
                # Fallback maybe already array natively stored
                img_rgb = np.array(img_obj)
                if img_rgb.shape[0] <= 4:
                    img_rgb = np.transpose(img_rgb, (1, 2, 0))
                    
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            if writer is None:
                fh, fw = img_bgr.shape[:2]
                if fw < 400:
                    fw, fh = fw * 2, fh * 2
                writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*'mp4v'), 30, (fw, fh))
            
            img_bgr = cv2.resize(img_bgr, (fw, fh))
            if action_key and frame_data.get(action_key) is not None:
                action_val = frame_data[action_key]
                val = action_val[0] if isinstance(action_val, (list, np.ndarray)) else action_val
                cv2.putText(img_bgr, f'Act[0]: {val:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(img_bgr, f'Ep: {ep_idx} | Frame: {t}/{n_frames}', (10, fh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            writer.write(img_bgr)
            
        if writer is not None:
            writer.release()
            
    print(f"\nHoàn tất (RAW Modal)! Video lưu tại: {out_dir}")

def explore_dataset(repo_id, num_episodes=5, output_dir="dataset_explorations"):
    print("=" * 70)
    print(f"  Khảo sát Dataset: {repo_id}")
    print("=" * 70)
    
    out_dir = Path(output_dir) / repo_id.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Đang tải qua lerobot...")
        dataset = LeRobotDataset(repo_id)
        _run_lerobot_dataset(dataset, out_dir, repo_id, num_episodes)
    except Exception as e:
        print(f"\n[CẢNH BÁO] LeRobotDataset không tương thích (Lỗi: {type(e).__name__}).")
        print("-> Tự động chuyển sang chế độ RAW HuggingFace datasets để phá rào hạn chế phiên bản...")
        try:
            _run_raw_hf_dataset(repo_id, out_dir, num_episodes)
        except Exception as raw_e:
            print(f"\n[LỖI NGHIÊM TRỌNG] Không thể tải bằng chế độ RAW: {raw_e}")


def main():
    parser = argparse.ArgumentParser(description="Script linh hoạt mở rộng để khảo sát đa dạng Dataset")
    parser.add_argument("--repo_id", type=str, default="bnarin/so100_tic_tac_toe_we_do_it_live",
                        help="Tên repo ID của dataset trên HF")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Số lượng samples/tình huống cần export ra video để đánh giá")
    parser.add_argument("--output_dir", type=str, default="d:/EyetechCode/dataset_explorations",
                        help="Thư mục xuất kết quả")
    
    args = parser.parse_args()
    
    if args.repo_id not in SUPPORTED_DATASETS:
        print(f"Lưu ý: Dataset '{args.repo_id}' không nằm trong danh sách mặc định bạn đã cung cấp, nhưng script vẫn sẽ thử tải nó.")

    explore_dataset(args.repo_id, args.num_episodes, args.output_dir)

if __name__ == "__main__":
    main()
