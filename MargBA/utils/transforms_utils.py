"""
Utility functions for saving camera poses and intrinsics to transforms.json format
Compatible with Nerfstudio and other NeRF frameworks
"""

import os
import json
import numpy as np
import torch
from pathlib import Path


def save_transforms_json(
    output_path,
    camera_poses,      # (N, 4, 4) OpenCV cam2world
    intrinsics,        # (N, 3, 3) or None
    image_paths,       # List[str], length N
    image_size,        # (H, W) tuple
):
    """
    Save camera poses and intrinsics to transforms.json (Nerfstudio format)
    
    Args:
        output_path: Path to save transforms.json
        camera_poses: (N, 4, 4) camera-to-world matrices in OpenCV convention
        intrinsics: (N, 3, 3) intrinsic matrices, or None
        image_paths: List of image file paths (relative)
        image_size: (H, W) tuple
    """
    N = camera_poses.shape[0]
    H, W = image_size
    
    # Convert OpenCV c2w (RDF: +x right, +y down, +z forward)
    # to OpenGL/Blender c2w (RUB: +x right, +y up, -z forward).
    camera_poses = np.array(camera_poses, copy=True)
    camera_poses[:, :3, 1:3] *= -1

    # Build transforms.json structure
    transforms_data = {
        "camera_model": "OpenGL",
    }
    
    # Check if all frames have the same intrinsics (shared camera mode)
    use_shared_camera = False
    if N > 1 and np.allclose(intrinsics[0], intrinsics[1:], rtol=1e-3):
        use_shared_camera = True
        K = intrinsics[0]
        transforms_data.update({
            "fl_x": float(K[0, 0]),
            "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "w": int(W),
            "h": int(H),
        })
    
    # Build frames
    frames = []
    for i in range(N):
        frame = {
            "file_path": image_paths[i],
            "transform_matrix": camera_poses[i].tolist()
        }
        
        # Add per-frame camera parameters if not using shared camera
        if not use_shared_camera:
            K = intrinsics[i]
            frame.update({
                "w": int(W),
                "h": int(H),
                "fl_x": float(K[0, 0]),
                "fl_y": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
            })
        
        frames.append(frame)
    
    transforms_data["frames"] = frames
    
    # Save to file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(transforms_data, f, indent=4)
    
    print(f"已保存 transforms.json ({N} 帧) 到: {output_path}")


def generate_image_paths(data_path, num_frames, save_dir='images'):
    """
    根据数据路径生成图像文件路径列表
    
    Args:
        data_path: 原始数据路径 (目录或视频文件)
        num_frames: 帧数
        save_dir: 保存图像的相对目录名
    
    Returns:
        image_paths: List[str] 相对路径列表
    """
    image_paths = []
    
    if os.path.isdir(data_path):
        # 从目录读取
        filenames = sorted([x for x in os.listdir(data_path) 
                          if x.lower().endswith((".png", ".jpg", ".jpeg", ".heic"))])
        for i in range(min(num_frames, len(filenames))):
            image_paths.append(f"./{save_dir}/{filenames[i]}")
    else:
        # 视频输入，生成帧文件名
        for i in range(num_frames):
            image_paths.append(f"./{save_dir}/frame_{i:04d}.png")
    
    return image_paths
