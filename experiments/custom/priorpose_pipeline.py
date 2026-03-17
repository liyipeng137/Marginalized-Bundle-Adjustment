import argparse
import copy
import glob
import json
import os
import shutil
import socket
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import h5py
import natsort
import numpy as np
import torch
import torch.multiprocessing as mp

PRJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PRJ_ROOT)

from third_party.colmap_read_write_model import read_model, qvec2rotmat
from MargBA.BA.bundle_adjustment import bundle_adjustment
from MargBA.datasets import HDF5Reader, HDF5Writer
from MargBA.corres_estimator import inference_pairwise
from MargBA.poses_models import GlobalOptimizationPoseParameters
from MargBA.utils.transforms_utils import save_transforms_json


def find_free_port(start_port=12362, end_port=13962, max_attempts=100):
    used_ports = set()
    attempt = 0
    while attempt < max_attempts:
        port = np.random.randint(start_port, end_port)
        if port in used_ports:
            continue
        used_ports.add(port)
        attempt += 1
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return str(port)
            except socket.error:
                continue
    raise RuntimeError(f"Could not find a free port after {max_attempts} attempts")


def normalize_path(p: str) -> str:
    return p.replace("\\", "/")


def detect_colmap_ext(colmap_model_path: str) -> str:
    bin_ok = all(
        os.path.exists(os.path.join(colmap_model_path, f"{k}.bin"))
        for k in ["cameras", "images", "points3D"]
    )
    txt_ok = all(
        os.path.exists(os.path.join(colmap_model_path, f"{k}.txt"))
        for k in ["cameras", "images", "points3D"]
    )
    if bin_ok:
        return ".bin"
    if txt_ok:
        return ".txt"
    raise FileNotFoundError(
        f"Cannot find COLMAP model files under {colmap_model_path}. "
        "Need cameras/images/points3D in .bin or .txt format."
    )


def canonical_rel_path(p: str) -> str:
    p = normalize_path(p).strip()
    while p.startswith("./"):
        p = p[2:]
    return p


def camera_to_intrinsic(camera) -> np.ndarray:
    model = camera.model
    p = camera.params
    if model in [
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
        "FOV",
        "THIN_PRISM_FISHEYE",
    ]:
        if len(p) < 3:
            raise ValueError(f"Camera model {model} has insufficient params: {len(p)}")
        fx = fy = float(p[0])
        cx, cy = float(p[1]), float(p[2])
    elif model in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"]:
        if len(p) < 4:
            raise ValueError(f"Camera model {model} has insufficient params: {len(p)}")
        fx, fy = float(p[0]), float(p[1])
        cx, cy = float(p[2]), float(p[3])
    else:
        raise NotImplementedError(f"Unsupported COLMAP camera model: {model}")

    K = np.eye(3, dtype=np.float32)
    K[0, 0], K[1, 1] = fx, fy
    K[0, 2], K[1, 2] = cx, cy
    return K


def image_to_w2c(image) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = qvec2rotmat(image.qvec).astype(np.float32)
    pose[:3, 3] = image.tvec.astype(np.float32)
    return pose


def collect_rgb_paths(data_root: str) -> List[str]:
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.JPG"]:
        image_paths.extend(glob.glob(os.path.join(data_root, ext)))
    image_paths = natsort.natsorted(image_paths)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No RGB images found under {data_root}")
    return image_paths


def build_depth_lookup(depth_root: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    depth_paths = natsort.natsorted(glob.glob(os.path.join(depth_root, "**", "*.png"), recursive=True))
    if len(depth_paths) == 0:
        raise FileNotFoundError(f"No depth png files found under {depth_root}")

    rel_lookup = dict()
    stem_lookup = dict()
    stem_duplicated = set()
    for dpath in depth_paths:
        rel = normalize_path(os.path.relpath(dpath, depth_root))
        rel_no_ext = os.path.splitext(rel)[0]
        rel_lookup[rel_no_ext] = dpath

        stem = os.path.splitext(os.path.basename(dpath))[0]
        if stem in stem_lookup:
            stem_duplicated.add(stem)
        else:
            stem_lookup[stem] = dpath

    for stem in stem_duplicated:
        stem_lookup.pop(stem, None)
    return rel_lookup, stem_lookup


def build_colmap_lookup(colmap_images: Dict) -> Tuple[Dict[str, object], Dict[str, object]]:
    exact_lookup = dict()
    base_lookup = dict()
    base_duplicated = set()
    for image in colmap_images.values():
        name = normalize_path(image.name)
        exact_lookup[name] = image
        base = os.path.basename(name)
        if base in base_lookup:
            base_duplicated.add(base)
        else:
            base_lookup[base] = image

    for base in base_duplicated:
        base_lookup.pop(base, None)
    return exact_lookup, base_lookup


def build_payload_from_colmap(
    rgb_paths: List[str],
    data_root: str,
    depth_root: str,
    cameras: Dict,
    images: Dict,
):
    colmap_exact, colmap_base = build_colmap_lookup(images)
    depth_rel_lookup, depth_stem_lookup = build_depth_lookup(depth_root)

    payload = []
    missing_colmap = []
    missing_depth = []

    for idx, rgb_path in enumerate(rgb_paths):
        rel_rgb = normalize_path(os.path.relpath(rgb_path, data_root))
        stem = os.path.splitext(os.path.basename(rel_rgb))[0]
        rel_no_ext = os.path.splitext(rel_rgb)[0]

        image_colmap = None
        if rel_rgb in colmap_exact:
            image_colmap = colmap_exact[rel_rgb]
        elif os.path.basename(rel_rgb) in colmap_base:
            image_colmap = colmap_base[os.path.basename(rel_rgb)]

        if image_colmap is None:
            missing_colmap.append(rel_rgb)
            continue

        depth_path = None
        if rel_no_ext in depth_rel_lookup:
            depth_path = depth_rel_lookup[rel_no_ext]
        elif stem in depth_stem_lookup:
            depth_path = depth_stem_lookup[stem]

        if depth_path is None:
            missing_depth.append(rel_rgb)
            continue

        camera = cameras[image_colmap.camera_id]
        K = camera_to_intrinsic(camera)
        w2c = image_to_w2c(image_colmap)
        payload.append(
            {
                "idx": idx,
                "rgb_path": rgb_path,
                "rgb_rel": rel_rgb,
                "file_path": rel_rgb,
                "depth_path": depth_path,
                "colmap_name": normalize_path(image_colmap.name),
                "K": K,
                "w2c": w2c,
            }
        )

    report = {
        "n_rgb": len(rgb_paths),
        "n_payload": len(payload),
        "missing_prior": missing_colmap,
        "missing_depth": missing_depth,
        "unused_prior": [],
    }
    return payload, report


def parse_intrinsic_from_transform_entry(frame: Dict, root: Dict) -> np.ndarray:
    fl_x = frame.get("fl_x", root.get("fl_x"))
    fl_y = frame.get("fl_y", root.get("fl_y"))
    cx = frame.get("cx", root.get("cx"))
    cy = frame.get("cy", root.get("cy"))
    if any(x is None for x in [fl_x, fl_y, cx, cy]):
        raise ValueError("fl_x/fl_y/cx/cy is missing in transforms.json")
    K = np.eye(3, dtype=np.float32)
    K[0, 0], K[1, 1] = float(fl_x), float(fl_y)
    K[0, 2], K[1, 2] = float(cx), float(cy)
    return K


def parse_w2c_from_transform_entry(frame: Dict) -> np.ndarray:
    if "transform_matrix" not in frame:
        raise ValueError("transform_matrix is missing in transforms frame")
    c2w_gl = np.asarray(frame["transform_matrix"], dtype=np.float32)
    if c2w_gl.shape != (4, 4):
        raise ValueError(f"transform_matrix must be 4x4, got {c2w_gl.shape}")
    # OpenGL c2w -> OpenCV c2w
    c2w_cv = np.array(c2w_gl, copy=True)
    c2w_cv[:3, 1:3] *= -1
    # OpenCV c2w -> OpenCV w2c
    w2c = np.linalg.inv(c2w_cv).astype(np.float32)
    return w2c


def build_transforms_lookup(frames: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    exact_lookup = dict()
    base_lookup = dict()
    base_duplicated = set()
    for frame in frames:
        if "file_path" not in frame:
            continue
        path = canonical_rel_path(frame["file_path"])
        exact_lookup[path] = frame
        base = os.path.basename(path)
        if base in base_lookup:
            base_duplicated.add(base)
        else:
            base_lookup[base] = frame
    for base in base_duplicated:
        base_lookup.pop(base, None)
    return exact_lookup, base_lookup


def build_payload_from_transforms(
    rgb_paths: List[str],
    data_root: str,
    depth_root: str,
    transforms_json_path: str,
):
    with open(transforms_json_path, "r") as f:
        transforms_data = json.load(f)
    frames = transforms_data.get("frames", [])
    if len(frames) == 0:
        raise ValueError(f"No frames found in {transforms_json_path}")

    frame_exact, frame_base = build_transforms_lookup(frames)
    depth_rel_lookup, depth_stem_lookup = build_depth_lookup(depth_root)

    payload = []
    missing_prior = []
    missing_depth = []
    used_frames = set()

    for idx, rgb_path in enumerate(rgb_paths):
        rel_rgb = normalize_path(os.path.relpath(rgb_path, data_root))
        stem = os.path.splitext(os.path.basename(rel_rgb))[0]
        rel_no_ext = os.path.splitext(rel_rgb)[0]

        frame = None
        if rel_rgb in frame_exact:
            frame = frame_exact[rel_rgb]
        elif os.path.basename(rel_rgb) in frame_base:
            frame = frame_base[os.path.basename(rel_rgb)]

        if frame is None:
            missing_prior.append(rel_rgb)
            continue

        try:
            K = parse_intrinsic_from_transform_entry(frame, transforms_data)
            w2c = parse_w2c_from_transform_entry(frame)
        except Exception as e:
            missing_prior.append(f"{rel_rgb} ({str(e)})")
            continue

        depth_path = None
        if rel_no_ext in depth_rel_lookup:
            depth_path = depth_rel_lookup[rel_no_ext]
        elif stem in depth_stem_lookup:
            depth_path = depth_stem_lookup[stem]
        if depth_path is None:
            missing_depth.append(rel_rgb)
            continue

        payload.append(
            {
                "idx": idx,
                "rgb_path": rgb_path,
                "rgb_rel": rel_rgb,
                "file_path": canonical_rel_path(frame.get("file_path", rel_rgb)),
                "depth_path": depth_path,
                "K": K,
                "w2c": w2c,
            }
        )
        used_frames.add(id(frame))

    unused_prior = []
    for frame in frames:
        if id(frame) not in used_frames:
            fp = frame.get("file_path", "<missing file_path>")
            unused_prior.append(canonical_rel_path(fp) if isinstance(fp, str) else str(fp))

    report = {
        "n_rgb": len(rgb_paths),
        "n_payload": len(payload),
        "missing_prior": missing_prior,
        "missing_depth": missing_depth,
        "unused_prior": unused_prior,
    }
    return payload, report


def fail_fast_report(report: Dict, prior_source_name: str):
    print("========== Data Contract Check ==========")
    print(f"RGB frames            : {report['n_rgb']}")
    print(f"Matched payload frames: {report['n_payload']}")
    print(f"Missing {prior_source_name} frames: {len(report['missing_prior'])}")
    print(f"Missing depth frames  : {len(report['missing_depth'])}")
    print(f"Unused {prior_source_name} frames : {len(report.get('unused_prior', []))}")
    if len(report["missing_prior"]) > 0:
        print(f"Examples missing {prior_source_name}:", report["missing_prior"][:10])
    if len(report["missing_depth"]) > 0:
        print("Examples missing depth :", report["missing_depth"][:10])
    if len(report.get("unused_prior", [])) > 0:
        print(f"Examples unused {prior_source_name}:", report["unused_prior"][:10])
    if len(report["missing_prior"]) > 0 or len(report["missing_depth"]) > 0:
        raise RuntimeError("Data contract check failed, please fix mapping before BA.")
    print("Data contract check passed.")


def prepare_hdf5_and_mapper(dst_perscene: str, scene: str, payload: List[Dict], overwrite: bool):
    os.makedirs(dst_perscene, exist_ok=True)
    h5path = os.path.join(dst_perscene, f"{scene}.hdf5")

    if overwrite and os.path.exists(h5path):
        os.remove(h5path)
    if (not overwrite) and os.path.exists(h5path):
        raise FileExistsError(
            f"{h5path} already exists. Use --overwrite to rebuild or --skip-preprocess to reuse."
        )

    writer = HDF5Writer(
        h5pypath=h5path,
        keys_to_add=["rgb", "depth_gt", "depth_pr", "intrinsic_gt", "pose_w2c_gt"],
        mode="w",
    )

    mapper_lines = []
    for item in payload:
        idx = item["idx"]
        idx_name = str(idx).zfill(6)
        writer.write_jpg_image(item["rgb_path"], f"{idx_name}.jpg")
        writer.write_depth_gt(item["depth_path"], f"{idx_name}.png")
        writer.write_depth_pr(item["depth_path"], f"{idx_name}.png")
        writer.hfw["intrinsic_gt"].create_dataset(f"{idx_name}.txt", data=item["K"])
        writer.hfw["pose_w2c_gt"].create_dataset(f"{idx_name}.txt", data=item["w2c"])
        mapper_lines.append(f"{idx_name} {item['rgb_rel']}\n")
    writer.hfw.close()

    mapper_path = os.path.join(dst_perscene, "mapper.txt")
    with open(mapper_path, "w") as f:
        f.writelines(mapper_lines)

    return h5path


def ensure_required_groups(h5path: str):
    required_groups = ["rgb", "corres_i2j", "visibility_i2j", "intrinsic_gt", "depth_gt"]
    with h5py.File(h5path, "r") as f:
        missing = [x for x in required_groups if x not in f]
    if len(missing) > 0:
        raise RuntimeError(f"HDF5 missing required groups: {missing}")


def adjust_intrinsic(intrinsic: torch.Tensor, depth_focalx: torch.Tensor, depth_focaly: torch.Tensor) -> torch.Tensor:
    fx_src, fy_src = intrinsic[:, 0, 0], intrinsic[:, 1, 1]
    fx_src_adj = torch.exp(torch.log(fx_src) + depth_focalx)
    fy_src_adj = torch.exp(torch.log(fy_src) + depth_focaly)
    intrinsic_adjusted = torch.clone(intrinsic)
    intrinsic_adjusted[:, 0, 0] = fx_src_adj
    intrinsic_adjusted[:, 1, 1] = fy_src_adj
    return intrinsic_adjusted


def prepare_prior_pose_init_ckpt(
    payload: List[Dict],
    sfm_perscene: str,
    sharefocal: bool,
    calibrated: bool,
):
    os.makedirs(sfm_perscene, exist_ok=True)
    nfrm = len(payload)
    intrinsics = np.stack([x["K"] for x in payload], axis=0)
    poses_w2c = np.stack([x["w2c"] for x in payload], axis=0)

    intrinsic_init = intrinsics[0] if sharefocal else intrinsics
    pr_pose_model = GlobalOptimizationPoseParameters(
        nfrm=nfrm,
        optimizefocal=(not calibrated),
        sharefocal=sharefocal,
        intrinsic=intrinsic_init,
    )
    for i in range(nfrm):
        pr_pose_model.update_pose_adjustment(
            nodeid=i,
            newpose=torch.from_numpy(poses_w2c[i]).float(),
            newadjustment=1.0,
            newbias=0.0,
        )
    torch.save(
        pr_pose_model.cpu().state_dict(),
        os.path.join(sfm_perscene, "pr_pose_model_init.ckpt"),
    )


def export_optimized_transforms(
    h5path: str,
    sfm_perscene: str,
    payload: List[Dict],
    sharefocal: bool,
    output_transforms_json: str,
):
    nfrm = len(payload)
    fine_ckpt = os.path.join(sfm_perscene, "pr_pose_model_fine.ckpt")
    if not os.path.exists(fine_ckpt):
        raise FileNotFoundError(f"Fine stage checkpoint not found: {fine_ckpt}")

    pose_model = GlobalOptimizationPoseParameters(
        nfrm=nfrm,
        optimizefocal=False,
        sharefocal=sharefocal,
    )
    pose_model.load_state_dict(torch.load(fine_ckpt), strict=True)

    pose_w2c = pose_model.get_pose_w2c().cpu().detach()
    pose_c2w = torch.inverse(pose_w2c)
    depth_adj, depth_bias, depth_focalx, depth_focaly = pose_model.get_depth_adjustment()
    del depth_adj, depth_bias
    intrinsic = pose_model.get_intrinsic().cpu().detach()
    intrinsic = adjust_intrinsic(
        intrinsic=intrinsic,
        depth_focalx=depth_focalx.cpu().detach(),
        depth_focaly=depth_focaly.cpu().detach(),
    )

    payload_sorted = sorted(payload, key=lambda x: x["idx"])
    image_paths = [x["file_path"] for x in payload_sorted]

    h5reader = HDF5Reader(h5path)
    img0 = h5reader.read_jpg_image("000000.jpg")
    w, h = img0.size
    save_transforms_json(
        output_path=output_transforms_json,
        camera_poses=pose_c2w.numpy(),
        intrinsics=intrinsic.numpy(),
        image_paths=image_paths,
        image_size=(h, w),
    )


def distribute_pairs_for_coarse(connections: List[List[int]], world_size: int):
    pair_qry_node = defaultdict(list)
    for idx1, idx2 in connections:
        pair_qry_node[idx1].append([idx1, idx2])
        pair_qry_node[idx2].append([idx1, idx2])

    per_gpu_pair = [[] for _ in range(world_size)]
    per_gpu_assigned_node = [[] for _ in range(world_size)]
    for node in pair_qry_node:
        lens = [len(x) for x in per_gpu_pair]
        to_assign_idx = int(np.argmin(np.array(lens)))
        per_gpu_pair[to_assign_idx].extend(pair_qry_node[node])
        per_gpu_assigned_node[to_assign_idx].append(node)
    return per_gpu_pair, per_gpu_assigned_node


def build_weight_dict(per_gpu_pair: List[List[List[int]]], coarse: bool):
    out = []
    for gpuid, gpu_pairs in enumerate(per_gpu_pair):
        weight_per_pair = dict()
        for i1, i2 in gpu_pairs:
            for srcid, dstid in [[i1, i2], [i2, i1]]:
                if coarse:
                    weight_per_pair[(srcid, dstid)] = np.ones(2)
                else:
                    weight_per_pair[(srcid, dstid)] = np.ones(1)
        out.append(weight_per_pair)
    return out


def run_ba(
    params_base: Dict,
    hdf5reader: HDF5Reader,
    nfrm: int,
    world_size: int,
    estimate_constant: int = 3500000,
):
    connections = hdf5reader.get_all_image_pairs()
    connections = [[int(x.split("_")[0]), int(x.split("_")[1])] for x in connections]
    if len(connections) == 0:
        raise RuntimeError("No correspondence pairs in HDF5, cannot run BA.")

    marker_query_map = ["qry"] * nfrm
    gradient_mask = np.ones(nfrm)

    # coarse
    per_gpu_pair, _ = distribute_pairs_for_coarse(connections, world_size)
    per_gpu_pair = [x[0 : min(params_base["maxpair"], len(x))] for x in per_gpu_pair]
    if not all(len(x) > 0 for x in per_gpu_pair):
        raise RuntimeError("At least one GPU got zero pairs in coarse stage.")
    per_gpu_weight_on_pair_dict = build_weight_dict(per_gpu_pair, coarse=True)

    nconnections = max([len(x) for x in per_gpu_pair])
    sample_num = int(min(params_base["per_gpu_ram"] / nconnections * estimate_constant, params_base["max_sample_num"]))
    sample_num = sample_num if sample_num % 2 == 0 else sample_num - 1

    params_coarse = copy.deepcopy(params_base)
    params_coarse.update(
        {
            "per_gpu_pair": per_gpu_pair,
            "marker_query_map": marker_query_map,
            "per_gpu_weight_on_pair_dict": per_gpu_weight_on_pair_dict,
            "stage": "coarse",
            "gradient_mask": gradient_mask,
            "lr_intrinsic_boost": 50,
            "sample_num": sample_num,
            "losses": {
                "cdf_log_subgraph": {"maxpx": 15.0, "bins": 250, "gradient_smooth": 2, "iterations": 50000}
            },
            "master_port": find_free_port(12362, 12962),
        }
    )
    mp.spawn(bundle_adjustment, args=(world_size, params_coarse), nprocs=world_size, join=True)

    # fine
    per_gpu_pair = [connections[i::world_size] for i in range(world_size)]
    per_gpu_pair = [x[0 : min(params_base["maxpair"], len(x))] for x in per_gpu_pair]
    if not all(len(x) > 0 for x in per_gpu_pair):
        raise RuntimeError("At least one GPU got zero pairs in fine stage.")
    per_gpu_weight_on_pair_dict = build_weight_dict(per_gpu_pair, coarse=False)

    nconnections = max([len(x) for x in per_gpu_pair])
    sample_num = int(min(params_base["per_gpu_ram"] / nconnections * estimate_constant, params_base["max_sample_num"]))
    sample_num = sample_num if sample_num % 2 == 0 else sample_num - 1

    params_fine = copy.deepcopy(params_base)
    params_fine.update(
        {
            "per_gpu_pair": per_gpu_pair,
            "marker_query_map": marker_query_map,
            "per_gpu_weight_on_pair_dict": per_gpu_weight_on_pair_dict,
            "sample_num": sample_num,
            "stage": "fine",
            "gradient_mask": gradient_mask,
            "lr_intrinsic_boost": 10,
            "losses": {
                "cdf_log": {"maxpx": 15.0, "bins": 250, "gradient_smooth": 2, "iterations": 10000},
                "cdf_euclidean": {"maxpx": 50.0, "bins": 600, "gradient_smooth": 3, "iterations": 10000},
            },
            "master_port": find_free_port(13000, 13600),
        }
    )
    mp.spawn(bundle_adjustment, args=(world_size, params_fine), nprocs=world_size, join=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Prior-pose + LiDAR-depth full pipeline for custom MBA")
    parser.add_argument("--data-root", type=str, required=True, help="RGB image folder (flat folder like custom dataset)")
    parser.add_argument("--depth-root", type=str, required=True, help="LiDAR depth folder (.png)")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--prior-pose-colmap-model", type=str, default=None, help="COLMAP model folder")
    source_group.add_argument("--input-transforms-json", type=str, default=None, help="Input transforms.json (OpenGL c2w)")
    parser.add_argument("--output-location", type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/custom")
    parser.add_argument("--output-transforms-json", type=str, default=None, help="Output optimized transforms.json path")
    # parser.add_argument("--depth-model", type=str, choices=["ZoeDepth", "UniDepth", "DUSt3R"], default="DUSt3R")
    parser.add_argument("--corres-model", type=str, choices=["RoMa", "MASt3R"], default="RoMa")
    parser.add_argument("--depth-source", type=str, choices=["marker", "mixed", "pr", "gt"], default="gt")
    parser.add_argument("--min-confidence", type=float, default=0.2, help="Dense matcher confidence threshold")
    parser.add_argument("--min-visibility", type=float, default=0.1, help="Dense matcher visibility threshold")
    parser.add_argument("--min-corres-conf", type=float, default=0.0, help="BA sampling confidence threshold")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing preprocessed artifacts")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip correspondences preprocessing and reuse existing hdf5")
    parser.add_argument("--sharefocal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibrated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--per-gpu-ram", type=float, default=32.0)
    parser.add_argument("--maxpair", type=float, default=np.inf)
    parser.add_argument("--max-sample-num", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    available_gpus = torch.cuda.device_count()
    if available_gpus < 1:
        raise RuntimeError("No CUDA device found. This pipeline requires GPU for dense matching and BA.")

    # Keep hdf5 scene naming consistent with corres preprocessing internals
    # (inference_pairwise uses os.path.basename(data_root) as scene key).
    scene = os.path.basename(os.path.normpath(args.data_root))
    if scene == "":
        raise ValueError(f"Cannot infer scene name from data_root: {args.data_root}")
    preprocess_location = os.path.join(args.output_location, f"{args.depth_model}_{args.corres_model}")
    dst_perscene = preprocess_location
    sfm_perscene = os.path.join(f"{preprocess_location}_sfm")
    os.makedirs(dst_perscene, exist_ok=True)
    os.makedirs(sfm_perscene, exist_ok=True)

    rgb_paths = collect_rgb_paths(args.data_root)
    if args.prior_pose_colmap_model is not None:
        colmap_ext = detect_colmap_ext(args.prior_pose_colmap_model)
        cameras, images, _ = read_model(args.prior_pose_colmap_model, colmap_ext)
        payload, report = build_payload_from_colmap(
            rgb_paths=rgb_paths,
            data_root=args.data_root,
            depth_root=args.depth_root,
            cameras=cameras,
            images=images,
        )
        fail_fast_report(report, prior_source_name="COLMAP")
    else:
        payload, report = build_payload_from_transforms(
            rgb_paths=rgb_paths,
            data_root=args.data_root,
            depth_root=args.depth_root,
            transforms_json_path=args.input_transforms_json,
        )
        fail_fast_report(report, prior_source_name="transforms")

    h5path = os.path.join(dst_perscene, f"{scene}.hdf5")
    if not args.skip_preprocess:
        if args.overwrite:
            if os.path.exists(os.path.join(dst_perscene, "intermediate")):
                shutil.rmtree(os.path.join(dst_perscene, "intermediate"))
            if os.path.exists(os.path.join(dst_perscene, "vls")):
                shutil.rmtree(os.path.join(dst_perscene, "vls"))
        h5path = prepare_hdf5_and_mapper(dst_perscene=dst_perscene, scene=scene, payload=payload, overwrite=args.overwrite)
        inference_pairwise(
            data_root=args.data_root,
            dataset="custom",
            output_location=dst_perscene,
            corres_method_name=args.corres_model,
            min_confidence=args.min_confidence,
            min_visibility=args.min_visibility,
        )
    else:
        if not os.path.exists(h5path):
            raise FileNotFoundError(f"{h5path} not found. --skip-preprocess requires existing hdf5.")
        mapper_path = os.path.join(dst_perscene, "mapper.txt")
        if not os.path.exists(mapper_path):
            with open(mapper_path, "w") as f:
                f.writelines([f"{str(x['idx']).zfill(6)} {x['rgb_rel']}\n" for x in payload])

    ensure_required_groups(h5path)
    prepare_prior_pose_init_ckpt(
        payload=payload,
        sfm_perscene=sfm_perscene,
        sharefocal=args.sharefocal,
        calibrated=args.calibrated,
    )

    hdf5reader = HDF5Reader(h5path)
    nfrm = len(hdf5reader.get_all_images())
    if nfrm != len(payload):
        raise RuntimeError(f"Frame mismatch: hdf5 has {nfrm} frames but payload has {len(payload)}")

    connections = hdf5reader.get_all_image_pairs()
    unique_nodes = set()
    for pair_name in connections:
        i1, i2 = pair_name.split("_")
        unique_nodes.add(int(i1))
        unique_nodes.add(int(i2))
    world_size = min(available_gpus, max(1, len(connections)), max(1, len(unique_nodes)))
    params_base = {
        "data_root": args.data_root,
        "lr": args.lr,
        "log_freq": 500,
        "max_pair_residual": 40,
        "min_corres_conf": args.min_corres_conf,
        "wt_gt": True,
        "evaluate_function": None,
        "twoview_ransac_th": 0.1,
        "camera_model": "SIMPLE_PINHOLE",
        "sharefocal": args.sharefocal,
        "calibrated": args.calibrated,
        "read_init_pose": True,
        "mapfree": False,
        "maxpair": args.maxpair,
        "per_gpu_ram": args.per_gpu_ram,
        "max_sample_num": args.max_sample_num,
        "preprocess_location": dst_perscene,
        "sfm_location": sfm_perscene,
        "hdf5name": scene,
        "nfrm": nfrm,
        "depth_source": args.depth_source,
    }

    run_ba(params_base=params_base, hdf5reader=hdf5reader, nfrm=nfrm, world_size=world_size)
    if args.output_transforms_json is None:
        output_transforms_json = os.path.join(sfm_perscene, "optimized_transforms.json")
    else:
        output_transforms_json = args.output_transforms_json
    export_optimized_transforms(
        h5path=h5path,
        sfm_perscene=sfm_perscene,
        payload=payload,
        sharefocal=args.sharefocal,
        output_transforms_json=output_transforms_json,
    )
    print("Pipeline finished.")
    print(f"HDF5: {h5path}")
    print(f"Init pose ckpt: {os.path.join(sfm_perscene, 'pr_pose_model_init.ckpt')}")
    print(f"Fine pose ckpt: {os.path.join(sfm_perscene, 'pr_pose_model_fine.ckpt')}")
    print(f"Optimized transforms: {output_transforms_json}")


if __name__ == "__main__":
    main()
