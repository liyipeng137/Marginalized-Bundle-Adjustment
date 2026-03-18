import importlib
import os
import sys

import numpy as np
import torch
from PIL import Image


def _purge_modules(prefixes):
    for name in list(sys.modules.keys()):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            sys.modules.pop(name, None)


def _load_mast3r_fast_modules():
    prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mast3r_fast_root = os.path.join(prj_root, "third_party", "mast3r-fast")
    if mast3r_fast_root not in sys.path:
        sys.path.insert(0, mast3r_fast_root)

    # `mast3r-fast` ships its own `mast3r` / `dust3r` packages.
    # Purge any previously imported versions so this backend uses the intended codepath.
    _purge_modules(["mast3r", "dust3r", "dust3r_visloc"])

    mast3r_model = importlib.import_module("mast3r.model")
    dust3r_transforms = importlib.import_module("dust3r.datasets.utils.transforms")
    return mast3r_model.AsymmetricMASt3R, dust3r_transforms.ImgNorm


def _coords_to_normalized(coords, height, width):
    coords = torch.from_numpy(coords).float()
    coords[:, 0] = (coords[:, 0] + 0.5) / width * 2 - 1
    coords[:, 1] = (coords[:, 1] + 0.5) / height * 2 - 1
    return coords


def _rescale_to_original(coords, orig_shape, resized_shape):
    orig_h, orig_w = orig_shape
    resized_h, resized_w = resized_shape
    coords = coords.copy()
    coords[:, 0] = (coords[:, 0] + 0.5) * (orig_w / resized_w) - 0.5
    coords[:, 1] = (coords[:, 1] + 0.5) * (orig_h / resized_h) - 0.5
    return coords


class MASt3RFast:
    def __init__(self, min_confidence, min_visibility):
        self.AsymmetricMASt3R, self.ImgNorm = _load_mast3r_fast_modules()
        weights_path = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        self.corres_estimator = self.AsymmetricMASt3R.from_pretrained(weights_path).eval()

        self.image_size = 512
        self.subsample = 8
        self.match_conf_thr = max(1.5, float(min_confidence))
        self.min_confidence = min_confidence
        self.min_visibility = min_visibility

    def to_cuda(self, deviceid):
        self.device = torch.device(f"cuda:{deviceid}")
        self.corres_estimator = self.corres_estimator.to(self.device)

    def _prepare_view(self, image_uint8):
        image = Image.fromarray(image_uint8).convert("RGB")
        orig_w, orig_h = image.size
        scale = min(self.image_size / max(orig_h, orig_w), 1.0)
        new_h = max((int(orig_h * scale) // 16) * 16, 16)
        new_w = max((int(orig_w * scale) // 16) * 16, 16)

        image_resized = image.resize((new_w, new_h), Image.LANCZOS)
        image_tensor = self.ImgNorm(image_resized).unsqueeze(0).to(self.device)
        return {
            "img": image_tensor,
            "true_shape": torch.tensor([[new_h, new_w]], dtype=torch.int32, device=self.device),
            "orig_shape": np.array([orig_h, orig_w], dtype=np.int32),
            "resized_shape": np.array([new_h, new_w], dtype=np.int32),
            "instance": "0",
        }

    @torch.no_grad()
    def _extract_correspondences_fast(self, desc1, desc2, conf1, conf2):
        h1, w1, dim = desc1.shape
        h2, w2, _ = desc2.shape

        ys = torch.arange(self.subsample // 2, h1, self.subsample, device=self.device)
        xs = torch.arange(self.subsample // 2, w1, self.subsample, device=self.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_y = grid_y.flatten()
        grid_x = grid_x.flatten()

        if grid_y.numel() == 0:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)), 0

        query_desc = desc1[grid_y, grid_x]
        query_conf = conf1[grid_y, grid_x]
        desc2_flat = desc2.reshape(-1, dim)

        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            dists = torch.cdist(query_desc.float(), desc2_flat.float())
            nn_idx = dists.argmin(dim=1)
            nn_y, nn_x = nn_idx // w2, nn_idx % w2
            matched_conf = conf2[nn_y, nn_x]
            combined_conf = torch.sqrt(torch.clamp_min(query_conf * matched_conf, 0.0))

        mask = combined_conf >= self.match_conf_thr
        matches_im0 = torch.stack([grid_x[mask].float(), grid_y[mask].float()], dim=1)
        matches_im1 = torch.stack([nn_x[mask].float(), nn_y[mask].float()], dim=1)
        return (
            matches_im0.cpu().numpy(),
            matches_im1.cpu().numpy(),
            combined_conf[mask].cpu().numpy(),
            int(grid_y.numel()),
        )

    def _rasterize_matches(self, matches_query, matches_map, matches_conf, query_shape, map_shape):
        query_h, query_w = query_shape
        map_h, map_w = map_shape
        warp = torch.zeros([query_h, query_w, 4], device=self.device)
        certainty = torch.zeros([1, query_h, query_w], device=self.device)

        if len(matches_conf) == 0:
            return warp, certainty

        qx = np.rint(matches_query[:, 0]).astype(np.int32)
        qy = np.rint(matches_query[:, 1]).astype(np.int32)
        valid = (
            (qx >= 0)
            & (qx < query_w)
            & (qy >= 0)
            & (qy < query_h)
            & (matches_map[:, 0] >= -0.5)
            & (matches_map[:, 0] < map_w - 0.5)
            & (matches_map[:, 1] >= -0.5)
            & (matches_map[:, 1] < map_h - 0.5)
        )
        if not np.any(valid):
            return warp, certainty

        matches_query = matches_query[valid]
        matches_map = matches_map[valid]
        matches_conf = matches_conf[valid]
        qx = qx[valid]
        qy = qy[valid]

        order = np.argsort(matches_conf)
        matches_query = matches_query[order]
        matches_map = matches_map[order]
        matches_conf = matches_conf[order]
        qx = qx[order]
        qy = qy[order]

        norm_query = _coords_to_normalized(matches_query, query_h, query_w).to(self.device)
        norm_map = _coords_to_normalized(matches_map, map_h, map_w).to(self.device)
        conf_tensor = torch.from_numpy(matches_conf).float().to(self.device)

        certainty[0, qy, qx] = conf_tensor
        warp[qy, qx, 0:2] = norm_query
        warp[qy, qx, 2:4] = norm_map
        return warp, certainty

    @torch.no_grad()
    def inference(self, data):
        _, _, query_h, query_w = data["rgb_src"].shape
        _, _, map_h, map_w = data["rgb_dst"].shape
        batch_size = data["rgb_src"].shape[0]

        warp_src_dst = torch.zeros([batch_size, query_h, query_w, 4], device=self.device)
        warp_dst_src = torch.zeros([batch_size, map_h, map_w, 4], device=self.device)
        certainty_src_dst = torch.zeros([batch_size, 1, query_h, query_w], device=self.device)
        certainty_dst_src = torch.zeros([batch_size, 1, map_h, map_w], device=self.device)
        visibility = torch.zeros([batch_size], device=self.device)

        for batch_idx in range(batch_size):
            rgb_src = data["rgb_src_wo_resize"][batch_idx].detach().cpu().numpy().astype(np.uint8)
            rgb_dst = data["rgb_dst_wo_resize"][batch_idx].detach().cpu().numpy().astype(np.uint8)

            view_src = self._prepare_view(rgb_src)
            view_dst = self._prepare_view(rgb_dst)
            pred1, pred2 = self.corres_estimator(view_src, view_dst)

            matches_src, matches_dst, conf_src_dst, nquery_src = self._extract_correspondences_fast(
                pred1["desc"][0], pred2["desc"][0], pred1["desc_conf"][0], pred2["desc_conf"][0]
            )
            matches_dst_rev, matches_src_rev, conf_dst_src, nquery_dst = self._extract_correspondences_fast(
                pred2["desc"][0], pred1["desc"][0], pred2["desc_conf"][0], pred1["desc_conf"][0]
            )

            matches_src = _rescale_to_original(matches_src, view_src["orig_shape"], view_src["resized_shape"])
            matches_dst = _rescale_to_original(matches_dst, view_dst["orig_shape"], view_dst["resized_shape"])
            matches_dst_rev = _rescale_to_original(matches_dst_rev, view_dst["orig_shape"], view_dst["resized_shape"])
            matches_src_rev = _rescale_to_original(matches_src_rev, view_src["orig_shape"], view_src["resized_shape"])

            warp01, certainty01 = self._rasterize_matches(
                matches_src, matches_dst, conf_src_dst, query_shape=(query_h, query_w), map_shape=(map_h, map_w)
            )
            warp10, certainty10 = self._rasterize_matches(
                matches_dst_rev, matches_src_rev, conf_dst_src, query_shape=(map_h, map_w), map_shape=(query_h, query_w)
            )

            warp_src_dst[batch_idx] = warp01
            warp_dst_src[batch_idx] = warp10
            certainty_src_dst[batch_idx] = certainty01
            certainty_dst_src[batch_idx] = certainty10

            vis01 = float(len(conf_src_dst)) / max(nquery_src, 1)
            vis10 = float(len(conf_dst_src)) / max(nquery_dst, 1)
            visibility[batch_idx] = 0.5 * (vis01 + vis10)

        valid_src_dst = certainty_src_dst > 0
        valid_dst_src = certainty_dst_src > 0
        return warp_src_dst, warp_dst_src, certainty_src_dst, certainty_dst_src, visibility, valid_src_dst, valid_dst_src
