"""
dino_v4_frozen_for_seno.py — Modul Terkunci A -> C (Hafizh -> Seno)

Berisi fungsi preprocessing, wrapper DINOv3/DINOv2 offline (weights/), dan ekstraktor
yang IDENTIK 1:1 dengan 04_v3_extract_pair_dino.ipynb.
Gunakan fungsi `extract_embeddings_for_seno(...)` di 01_model.ipynb dan 02_submission.ipynb.
"""
import gc
import os
import json
import time
import random
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    import timm
except ImportError:
    timm = None
try:
    import transformers
except ImportError:
    transformers = None


# ==================== KONFIGURASI TERKUNCI (V4 CONTRACT) ====================
SEED = 42
BACKBONE_CHOICE = "dinov3_sat_large"   # Pilihan: "dinov3_sat_large", "dinov2_large", "dinov2_base"
IMG_SIZE_V3 = 256                      # Kelipatan 16 -> 16x16 = 256 patch untuk DINOv3 SAT-493M
DINOV2_FALLBACK_IMG_SIZE = 308         # Kelipatan 14 -> 22x22 = 484 patch untuk DINOv2-L / DINOv2-B
BATCH_SIZE = 16
NUM_WORKERS = 2
EXTRACT_GRAY3 = True
USE_FP16 = True                        # Sesuai keputusan user: FP16 autocast (sama persis di extractor & submission)
ALLOW_DATAPARALLEL = False             # Default 1 GPU agar konsisten dengan 1x Kaggle T4 di 02_submission.ipynb
TINT_NORM_THR = 0.05                   # Pure asymmetric max(0, min(R-G, B-G))
SHARED_CROP_MIN_RATIO = 0.50           # Side shared jika min/max >= 0.50
CLOUD_PATCH_THR = 0.50                 # Maksimal fraksi awan per patch untuk pooling post/changed
# ===========================================================================


def set_reproducible_seed(seed: int = SEED):
    """Kunci RNG dan flag deterministik PyTorch/cuDNN agar rerun konsisten."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_hf_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if tok:
        return tok.strip()
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            return tok.strip()
    except Exception:
        pass
    try:
        from google.colab import userdata
        tok = userdata.get("HF_TOKEN")
        if tok:
            return tok.strip()
    except Exception:
        pass
    return None


def detect_valid_mask(rgb: np.ndarray, sum_thr: int = 10) -> np.ndarray:
    """Deteksi area piksel non-hitam valid."""
    return rgb.sum(axis=2) > sum_thr


def _extract_valid_crop(rgb: np.ndarray):
    valid_raw = detect_valid_mask(rgb)
    if int(valid_raw.sum()) < 50:
        return rgb.copy(), np.ones(rgb.shape[:2], dtype=bool), max(rgb.shape[:2]), False
    ys, xs = np.where(valid_raw)
    y0, y1 = max(0, int(ys.min()) - 4), min(rgb.shape[0], int(ys.max()) + 5)
    x0, x1 = max(0, int(xs.min()) - 4), min(rgb.shape[1], int(xs.max()) + 5)
    crop = rgb[y0:y1, x0:x1].copy()
    m_crop = valid_raw[y0:y1, x0:x1]
    side = max(crop.shape[0], crop.shape[1])
    return crop, m_crop, side, True


def _pad_inpaint_resize(crop: np.ndarray, m_crop: np.ndarray, side: int, out_size: int):
    h, w = crop.shape[:2]
    side = max(side, h, w)
    py, px = (side - h) // 2, (side - w) // 2
    sq_img = np.zeros((side, side, 3), dtype=np.uint8)
    sq_msk = np.zeros((side, side), dtype=bool)
    sq_img[py:py + h, px:px + w] = crop
    sq_msk[py:py + h, px:px + w] = m_crop

    if not sq_msk.all() and sq_msk.any():
        _, idx = distance_transform_edt(~sq_msk, return_indices=True)
        filled = sq_img.copy()
        filled[~sq_msk] = sq_img[idx[0][~sq_msk], idx[1][~sq_msk]]
        blurred = cv2.GaussianBlur(filled, (15, 15), 0)
        sq_img[~sq_msk] = blurred[~sq_msk]

    interp = cv2.INTER_AREA if side >= out_size else cv2.INTER_LINEAR
    return (
        cv2.resize(sq_img, (out_size, out_size), interpolation=interp),
        cv2.resize(sq_msk.astype(np.float32), (out_size, out_size), interpolation=cv2.INTER_AREA),
    )


def crop_pad_inpaint(rgb: np.ndarray, out_size: int, force_side: int = None):
    """Crop ketat bounding box valid, pad persegi (opsional shared side), dan inpaint tepi kosong."""
    crop, m_crop, side, ok = _extract_valid_crop(rgb)
    if not ok:
        return (
            cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA),
            np.ones((out_size, out_size), dtype=np.float32),
        )
    use_side = side if force_side is None else max(force_side, side)
    return _pad_inpaint_resize(crop, m_crop, use_side, out_size)


def crop_pad_inpaint_pair(rgb_pre: np.ndarray, rgb_post: np.ndarray, out_size: int):
    """Gunakan side_shared jika rasio skala pre/post >= 0.50 agar resolusi spasial seragam."""
    cp, mp, sp, ok_p = _extract_valid_crop(rgb_pre)
    cq, mq, sq, ok_q = _extract_valid_crop(rgb_post)
    if not ok_p or not ok_q:
        return crop_pad_inpaint(rgb_pre, out_size), crop_pad_inpaint(rgb_post, out_size), False

    ratio = min(sp, sq) / float(max(sp, sq) + 1e-6)
    if ratio >= SHARED_CROP_MIN_RATIO:
        side_shared = min(1024, max(sp, sq))
        return _pad_inpaint_resize(cp, mp, side_shared, out_size), _pad_inpaint_resize(cq, mq, side_shared, out_size), True
    return _pad_inpaint_resize(cp, mp, sp, out_size), _pad_inpaint_resize(cq, mq, sq, out_size), False


def selective_color_norm(rgb_sq: np.ndarray, msk_sq: np.ndarray, tint_thr: float = TINT_NORM_THR) -> np.ndarray:
    """Detektor tint asimetris murni (R > G AND B > G) pada skala [0, 255]; pertahankan hutan/laut alami."""
    arr = rgb_sq.astype(np.float32)
    v = msk_sq > 0.5
    if int(v.sum()) < 30:
        v = np.ones_like(msk_sq, dtype=bool)
    rm, gm, bm = arr[:, :, 0][v].mean(), arr[:, :, 1][v].mean(), arr[:, :, 2][v].mean()
    tint = max(0.0, min(float(rm - gm), float(bm - gm))) / 255.0
    if tint > tint_thr:
        mu = np.array([rm, gm, bm], dtype=np.float32)[None, None, :]
        sd = np.maximum(np.array([arr[:, :, c][v].std() for c in range(3)], dtype=np.float32), 12.0)[None, None, :]
        return np.clip(127.5 + 48.0 * ((arr - mu) / sd), 0.0, 255.0).astype(np.uint8)
    return rgb_sq


def make_gray3_clahe(rgb_sq: np.ndarray) -> np.ndarray:
    """Cabang grayscale CLAHE 3-channel yang kebal acak warna/tint."""
    g = cv2.cvtColor(rgb_sq, cv2.COLOR_RGB2GRAY)
    cg = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return np.stack([cg, cg, cg], axis=-1)


class PatchTokenWrapper(nn.Module):
    """Ekstrak patch spasial murni dengan membuang prefix token (CLS + register tokens) secara dinamis."""
    def __init__(self, core_model, is_hf: bool, n_prefix: int):
        super().__init__()
        self.core_model = core_model
        self.is_hf = is_hf
        self.n_prefix = n_prefix

    def forward(self, x):
        if self.is_hf:
            tokens = self.core_model(pixel_values=x).last_hidden_state
        else:
            tokens = self.core_model.forward_features(x)
        return tokens[:, self.n_prefix:, :]


def _has_hf_weights(d: str) -> bool:
    return bool(d) and os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")) and (
        os.path.exists(os.path.join(d, "model.safetensors")) or os.path.exists(os.path.join(d, "pytorch_model.bin"))
    )


def _has_timm_weights(d: str) -> bool:
    return bool(d) and os.path.isdir(d) and os.path.exists(os.path.join(d, "model_state.pt")) and os.path.exists(os.path.join(d, "backbone_config.json"))


def find_local_weights_dir(backbone_choice: str = BACKBONE_CHOICE, explicit_dir: str = None):
    """Cari folder bobot lokal di ./weights, /kaggle/working/weights, atau /kaggle/input/ agar bisa jalan offline."""
    candidates = []
    if explicit_dir:
        candidates.append(explicit_dir)
    for base in ["./weights", "/kaggle/working/weights", "/kaggle/input"]:
        if not os.path.exists(base):
            continue
        candidates.append(os.path.join(base, backbone_choice))
        if base == "/kaggle/input":
            for root, dirs, _ in os.walk(base):
                if os.path.basename(root) == backbone_choice:
                    candidates.append(root)
                if "weights" in dirs:
                    candidates.append(os.path.join(root, "weights", backbone_choice))
    for cand in candidates:
        if _has_hf_weights(cand) or _has_timm_weights(cand):
            return cand
    return None


def _build_meta_info(base_info: dict, device: str, loaded_from: str):
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    base_info.update({
        "seed": SEED,
        "tint_norm_thr": TINT_NORM_THR,
        "cloud_pool_thr": CLOUD_PATCH_THR,
        "shared_crop_min_ratio": SHARED_CROP_MIN_RATIO,
        "precision": "fp16_autocast" if (USE_FP16 and device == "cuda") else "fp32",
        "device": device,
        "gpu_name": gpu_name,
        "n_gpus_visible": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "loaded_from": loaded_from,
        "torch_version": torch.__version__,
        "transformers_version": getattr(transformers, "__version__", "not_installed"),
        "timm_version": getattr(timm, "__version__", "not_installed"),
    })
    return base_info


def _save_backbone_meta(save_dir: str, info: dict):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "backbone_config.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def load_dino_backbone(
    backbone_choice: str = BACKBONE_CHOICE,
    device: str = None,
    hf_token: str = None,
    local_weights_dir: str = None,
    save_weights_dir: str = None,
):
    """Muat backbone secara offline dari weights/ bila tersedia; jika online, download lalu simpan ke save_weights_dir."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if hf_token is None:
        hf_token = get_hf_token()

    found_local = find_local_weights_dir(backbone_choice, explicit_dir=local_weights_dir)
    if found_local:
        if _has_hf_weights(found_local):
            from transformers import AutoImageProcessor, AutoModel
            proc = AutoImageProcessor.from_pretrained(found_local, local_files_only=True)
            hf_m = AutoModel.from_pretrained(found_local, local_files_only=True).eval().to(device)
            n_pref = 1 + int(getattr(hf_m.config, "num_register_tokens", 4 if "dinov3" in backbone_choice else 0))
            p_sz = int(getattr(hf_m.config, "patch_size", 16 if "dinov3" in backbone_choice else 14))
            img_sz = IMG_SIZE_V3 if "dinov3" in backbone_choice else DINOV2_FALLBACK_IMG_SIZE
            cfg_p = os.path.join(found_local, "backbone_config.json")
            saved_cfg = json.load(open(cfg_p, "r", encoding="utf-8")) if os.path.exists(cfg_p) else {}
            info = _build_meta_info({
                "model": saved_cfg.get("model", found_local),
                "backbone_choice": backbone_choice,
                "patch_size": int(saved_cfg.get("patch_size", p_sz)),
                "n_prefix": int(saved_cfg.get("n_prefix", n_pref)),
                "img_size": int(saved_cfg.get("img_size", img_sz)),
                "dim": int(hf_m.config.hidden_size),
                "mean": list(getattr(proc, "image_mean", saved_cfg.get("mean", [0.430, 0.411, 0.296]))),
                "std": list(getattr(proc, "image_std", saved_cfg.get("std", [0.213, 0.156, 0.143]))),
                "license": saved_cfg.get("license", "DINOv3 License" if "dinov3" in backbone_choice else "Apache-2.0"),
            }, device=device, loaded_from=f"offline:{found_local}")
            print(f"[OFFLINE OK] Model dimuat lokal dari {found_local}")
            return PatchTokenWrapper(hf_m, True, info["n_prefix"]), info

        if _has_timm_weights(found_local):
            saved_cfg = json.load(open(os.path.join(found_local, "backbone_config.json"), "r", encoding="utf-8"))
            m = timm.create_model(saved_cfg["model"], pretrained=False, num_classes=0, img_size=saved_cfg["img_size"]).eval()
            state = torch.load(os.path.join(found_local, "model_state.pt"), map_location="cpu")
            m.load_state_dict(state)
            m = m.to(device)
            info = _build_meta_info(saved_cfg, device=device, loaded_from=f"offline:{found_local}")
            print(f"[OFFLINE OK] Model timm dimuat lokal dari {found_local}")
            return PatchTokenWrapper(m, False, info["n_prefix"]), info

    if backbone_choice == "dinov3_sat_large":
        try:
            from transformers import AutoImageProcessor, AutoModel
            repo = "facebook/dinov3-vitl16-pretrain-sat493m"
            proc = AutoImageProcessor.from_pretrained(repo, token=hf_token)
            hf_m = AutoModel.from_pretrained(repo, token=hf_token).eval().to(device)
            n_pref = 1 + int(getattr(hf_m.config, "num_register_tokens", 4))
            p_sz = int(getattr(hf_m.config, "patch_size", 16))
            info = _build_meta_info({
                "model": repo,
                "backbone_choice": "dinov3_sat_large",
                "patch_size": p_sz,
                "n_prefix": n_pref,
                "img_size": IMG_SIZE_V3,
                "dim": int(hf_m.config.hidden_size),
                "mean": list(getattr(proc, "image_mean", [0.430, 0.411, 0.296])),
                "std": list(getattr(proc, "image_std", [0.213, 0.156, 0.143])),
                "license": "DINOv3 License",
            }, device=device, loaded_from=f"online:{repo}")
            if save_weights_dir:
                out_w = os.path.join(save_weights_dir, "dinov3_sat_large")
                os.makedirs(out_w, exist_ok=True)
                proc.save_pretrained(out_w)
                hf_m.save_pretrained(out_w)
                _save_backbone_meta(out_w, info)
                print(f"[SAVED WEIGHTS] Bobot DINOv3 disimpan untuk offline: {out_w}")
            return PatchTokenWrapper(hf_m, True, n_pref), info
        except Exception as e:
            print(f"[INFO] Fallback ke DINOv2 karena: {type(e).__name__}: {str(e)[:160]}")

    if backbone_choice == "dinov2_base":
        from transformers import AutoImageProcessor, AutoModel
        repo = "facebook/dinov2-base"
        proc = AutoImageProcessor.from_pretrained(repo)
        hf_m = AutoModel.from_pretrained(repo).eval().to(device)
        n_pref = 1 + int(getattr(hf_m.config, "num_register_tokens", 0))
        info = _build_meta_info({
            "model": repo,
            "backbone_choice": "dinov2_base",
            "patch_size": 14,
            "n_prefix": n_pref,
            "img_size": DINOV2_FALLBACK_IMG_SIZE,
            "dim": int(hf_m.config.hidden_size),
            "mean": list(getattr(proc, "image_mean", [0.485, 0.456, 0.406])),
            "std": list(getattr(proc, "image_std", [0.229, 0.224, 0.225])),
            "license": "Apache-2.0",
        }, device=device, loaded_from=f"online:{repo}")
        if save_weights_dir:
            out_w = os.path.join(save_weights_dir, "dinov2_base")
            os.makedirs(out_w, exist_ok=True)
            proc.save_pretrained(out_w)
            hf_m.save_pretrained(out_w)
            _save_backbone_meta(out_w, info)
            print(f"[SAVED WEIGHTS] Bobot DINOv2-Base disimpan untuk offline: {out_w}")
        return PatchTokenWrapper(hf_m, True, n_pref), info

    name = "vit_large_patch14_dinov2.lvd142m"
    m = timm.create_model(name, pretrained=True, num_classes=0, img_size=DINOV2_FALLBACK_IMG_SIZE).eval().to(device)
    n_pref = int(getattr(m, "num_prefix_tokens", 1))
    info = _build_meta_info({
        "model": name,
        "backbone_choice": "dinov2_large",
        "patch_size": 14,
        "n_prefix": n_pref,
        "img_size": DINOV2_FALLBACK_IMG_SIZE,
        "dim": int(m.num_features),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "license": "Apache-2.0",
    }, device=device, loaded_from=f"online:{name}")
    if save_weights_dir:
        out_w = os.path.join(save_weights_dir, "dinov2_large")
        os.makedirs(out_w, exist_ok=True)
        torch.save(m.state_dict(), os.path.join(out_w, "model_state.pt"))
        _save_backbone_meta(out_w, info)
        print(f"[SAVED WEIGHTS] Bobot DINOv2-Large disimpan untuk offline: {out_w}")
    return PatchTokenWrapper(m, False, n_pref), info


def d4_augment_batch(x: torch.Tensor, m: torch.Tensor, fl: bool, rot: int):
    """Terapkan salah satu dari 8 transformasi simetri D4 secara sinkron pada gambar & mask."""
    xf = torch.flip(x, [3]) if fl else x
    mf = torch.flip(m.float(), [2]) > 0.5 if fl else m
    xv = torch.rot90(xf, rot, [2, 3]) if rot > 0 else xf
    mv = torch.rot90(mf, rot, [1, 2]) if rot > 0 else mf
    return xv, mv


def extract_patch_grid(model, x_v: torch.Tensor, m_v: torch.Tensor, device: str = None, use_fp16: bool = USE_FP16):
    """Jalankan forward lalu kembalikan patch L2-normalized & pooled embedding ter-mask."""
    if device is None:
        device = str(x_v.device).split(":")[0]
    bs = x_v.shape[0]
    amp_enabled = bool(use_fp16 and device == "cuda")
    with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
        toks = model(x_v)
    toks_n = F.normalize(toks.float(), dim=2)
    mw = m_v.reshape(bs, -1, 1).float()
    pooled = F.normalize((toks_n * mw).sum(dim=1) / (mw.sum(dim=1) + 1e-6), dim=1)
    return toks_n, mw, pooled


def compute_spatial_stats(cmap: np.ndarray, vmask: np.ndarray, grid_sz: int):
    """Statistik spasial perubahan patch bebas rotasi berbasis P10 + P15 + floor absolut."""
    out = {}
    keys = (
        "frac_chg_abs", "frac_chg_p10", "chg_mean_p10", "chg_p90_p10",
        "frac_chg_p15", "chg_mean_p15", "chg_p90_p15", "massive_damage_guard",
        "spatial_entropy", "top10_share", "radial_decay", "com_shift", "axis_skew",
    )
    if int(vmask.sum()) < 4:
        for k in keys:
            out[k] = 0.0
        return out, np.ones(grid_sz * grid_sz, dtype=np.float32) / float(grid_sz * grid_sz)

    vals = cmap[vmask]
    p10 = float(np.percentile(vals, 10))
    p15 = float(np.percentile(vals, 15))
    crel10 = np.maximum(cmap - p10, 0.0) * vmask.astype(np.float32)
    crel15 = np.maximum(cmap - p15, 0.0) * vmask.astype(np.float32)
    vrel10 = crel10[vmask]
    vrel15 = crel15[vmask]

    out["frac_chg_abs"] = float(np.mean(vals > 0.25))
    out["frac_chg_p10"] = float(np.mean(vrel10 > 0.12))
    out["chg_mean_p10"] = float(vrel10.mean())
    out["chg_p90_p10"] = float(np.percentile(vrel10, 90))
    out["frac_chg_p15"] = float(np.mean(vrel15 > 0.12))
    out["chg_mean_p15"] = float(vrel15.mean())
    out["chg_p90_p15"] = float(np.percentile(vrel15, 90))
    out["massive_damage_guard"] = float((out["frac_chg_p15"] < 0.05) and (out["frac_chg_abs"] > 0.50))

    if out["massive_damage_guard"] > 0.5:
        crel_use = np.maximum(cmap - 0.10, 0.0) * vmask.astype(np.float32)
    else:
        crel_use = crel15
    vrel = crel_use[vmask]
    total = float(vrel.sum()) + 1e-8
    probs = vrel / total

    out["spatial_entropy"] = float(-np.sum(probs * np.log(probs + 1e-12)) / (np.log(len(probs) + 1e-12) + 1e-8))
    k10 = max(1, int(0.10 * len(vrel)))
    out["top10_share"] = float(np.sort(vrel)[-k10:].sum() / total)

    ys, xs = np.mgrid[0:grid_sz, 0:grid_sz].astype(np.float32)
    yv, xv = ys[vmask], xs[vmask]
    cy, cx = float(np.sum(probs * yv)), float(np.sum(probs * xv))
    out["com_shift"] = float(np.hypot(cy - (grid_sz - 1) / 2.0, cx - (grid_sz - 1) / 2.0) / (grid_sz * 0.5))

    pk = int(np.argmax(crel_use))
    py, px = float(pk // grid_sz), float(pk % grid_sz)
    dist = np.hypot(yv - py, xv - px)
    out["radial_decay"] = float(-np.corrcoef(dist, vrel)[0, 1]) if dist.std() > 1e-6 and vrel.std() > 1e-6 else 0.0

    coords = np.stack([xv - cx, yv - cy], axis=1)
    cov = (coords * probs[:, None]).T @ coords
    _, eigvecs = np.linalg.eigh(cov)
    proj = coords @ eigvecs[:, -1]
    sd_p = float(np.sqrt(np.sum(probs * (proj ** 2))) + 1e-6)
    out["axis_skew"] = float(abs(np.sum(probs * (proj ** 3)) / (sd_p ** 3)))

    weights = crel_use.reshape(-1)
    wsum = float(weights.sum())
    weights = (weights / wsum) if wsum > 1e-8 else (vmask.reshape(-1).astype(np.float32) / float(vmask.sum()))
    return out, weights.astype(np.float32)


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, mean, std, mode: str = "color_norm"):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.mode = mode
        self.mean = np.array(mean, dtype=np.float32)[:, None, None]
        self.std = np.array(std, dtype=np.float32)[:, None, None]

    def __len__(self):
        return len(self.df)

    def _to_tensor_and_cloud(self, sq_img: np.ndarray, sq_msk: np.ndarray):
        view = make_gray3_clahe(sq_img) if self.mode == "gray3" else selective_color_norm(sq_img, sq_msk)
        arr = sq_img.astype(np.float32) / 255.0
        bright = arr.mean(axis=2)
        cmax, cmin = arr.max(axis=2), arr.min(axis=2)
        sat = (cmax - cmin) / (cmax + 1e-5)
        cld = ((bright > 0.85) & (sat < 0.14)).astype(np.float32)

        chw = ((view.transpose(2, 0, 1).astype(np.float32) / 255.0) - self.mean) / self.std
        chw[:, sq_msk <= 0.5] = 0.0
        return torch.from_numpy(chw), torch.from_numpy(sq_msk), torch.from_numpy(cld)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        bgr_p, bgr_q = cv2.imread(r["pre_path"]), cv2.imread(r["post_path"])
        if bgr_p is None or bgr_q is None:
            raise FileNotFoundError(f"Gagal membaca gambar: {r['Id']}")
        (sq_p, m_p), (sq_q, m_q), _ = crop_pad_inpaint_pair(
            cv2.cvtColor(bgr_p, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(bgr_q, cv2.COLOR_BGR2RGB),
            self.img_size,
        )
        xp, mp, _ = self._to_tensor_and_cloud(sq_p, m_p)
        xq, mq, cq = self._to_tensor_and_cloud(sq_q, m_q)
        return xp, mp, xq, mq, cq


def select_cloud_excluded_mask(mq_grid: torch.Tensor, cq_grid: torch.Tensor):
    """Pool post/changed di atas patch bebas awan (<= CLOUD_PATCH_THR); fallback all_valid hanya jika <4 patch bersih."""
    vq_all = mq_grid >= 0.5
    vq_clean = vq_all & (cq_grid <= CLOUD_PATCH_THR)
    clean_cnt = vq_clean.reshape(vq_clean.shape[0], -1).sum(dim=1)
    fallback_flag = clean_cnt < 4
    vq_pool = torch.where(fallback_flag[:, None, None], vq_all, vq_clean)
    return vq_pool, vq_all, fallback_flag


def run_pairwise_extraction(
    model,
    df: pd.DataFrame,
    b_info: dict,
    mode: str = "color_norm",
    compute_patch_stats: bool = True,
    device: str = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    use_fp16: bool = USE_FP16,
):
    if device is None:
        device = b_info.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    img_sz, p_sz, emb_dim = b_info["img_size"], b_info["patch_size"], b_info["dim"]
    g_sz = img_sz // p_sz
    ds = PairDataset(df, img_sz, b_info["mean"], b_info["std"], mode=mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))

    N = len(df)
    ids = df["Id"].astype(str).tolist()
    emb_pre = np.zeros((N, emb_dim), dtype=np.float32)
    emb_post = np.zeros((N, emb_dim), dtype=np.float32)
    emb_chg = np.zeros((N, emb_dim), dtype=np.float32) if compute_patch_stats else None
    pstats = []
    ptr = 0

    with torch.no_grad():
        for xp, mp, xq, mq, cq in tqdm(dl, desc=f"Extract [{mode}]"):
            bs = xp.shape[0]
            xp, mp = xp.to(device, non_blocking=True), mp.to(device, non_blocking=True)
            xq, mq, cq = xq.to(device, non_blocking=True), mq.to(device, non_blocking=True), cq.to(device, non_blocking=True)

            vp_base = F.avg_pool2d(mp.unsqueeze(1), p_sz, p_sz).squeeze(1) >= 0.5
            mq_grid = F.avg_pool2d(mq.unsqueeze(1), p_sz, p_sz).squeeze(1)
            cq_grid = F.avg_pool2d(cq.unsqueeze(1), p_sz, p_sz).squeeze(1)
            vq_pool, vq_all, fb_flag = select_cloud_excluded_mask(mq_grid, cq_grid)

            acc_pre = torch.zeros((bs, emb_dim), device=device, dtype=torch.float32)
            acc_post = torch.zeros((bs, emb_dim), device=device, dtype=torch.float32)
            pre_bank = []
            post_tok0 = None

            for fl in (False, True):
                for rot in (0, 1, 2, 3):
                    xp_v, mp_v = d4_augment_batch(xp, vp_base, fl, rot)
                    xq_v, mq_v = d4_augment_batch(xq, vq_pool, fl, rot)

                    tp_n, mp_w, gp = extract_patch_grid(model, xp_v, mp_v, device=device, use_fp16=use_fp16)
                    tq_n, _, gq = extract_patch_grid(model, xq_v, mq_v, device=device, use_fp16=use_fp16)
                    acc_pre += gp
                    acc_post += gq

                    if compute_patch_stats:
                        pre_bank.append(tp_n * mp_w)
                        if not fl and rot == 0:
                            post_tok0 = tq_n

            emb_pre[ptr:ptr + bs] = F.normalize(acc_pre, dim=1).cpu().numpy()
            emb_post[ptr:ptr + bs] = F.normalize(acc_post, dim=1).cpu().numpy()

            if compute_patch_stats:
                bank_all = torch.cat(pre_bank, dim=1)
                sim = torch.bmm(post_tok0, bank_all.transpose(1, 2))
                max_sim, _ = sim.max(dim=2)
                chg_map = (1.0 - max_sim).reshape(bs, g_sz, g_sz).cpu().numpy()
                vp_np = vq_pool.cpu().numpy()
                va_np = vq_all.cpu().numpy()
                fb_np = fb_flag.cpu().numpy()
                pt_np = post_tok0.cpu().numpy()
                for i in range(bs):
                    st, w_chg = compute_spatial_stats(chg_map[i], vp_np[i], g_sz)
                    st_all, _ = compute_spatial_stats(chg_map[i], va_np[i], g_sz)
                    st["cloud_excluded_fallback_flag"] = float(fb_np[i])
                    for k_a, v_a in st_all.items():
                        st[f"{k_a}_allvalid"] = v_a
                    st["Id"] = ids[ptr + i]
                    pstats.append(st)
                    vec = (pt_np[i] * w_chg[:, None]).sum(axis=0)
                    emb_chg[ptr + i] = vec / (np.linalg.norm(vec) + 1e-8)

            ptr += bs

    return emb_pre, emb_post, emb_chg, pd.DataFrame(pstats) if compute_patch_stats else None


def compute_tes_a_retrieval(pre_emb: np.ndarray, post_emb: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Tes A: pencocokan post -> pre per pasangan (tanpa label), mengukur ketahanan preprocessing."""
    n = pre_emb.shape[0]
    p_norm = pre_emb / (np.linalg.norm(pre_emb, axis=1, keepdims=True) + 1e-8)
    q_norm = post_emb / (np.linalg.norm(post_emb, axis=1, keepdims=True) + 1e-8)
    sim = q_norm @ p_norm.T
    ranks = np.argsort(-sim, axis=1)
    targets = np.arange(n)[:, None]
    out = {}
    for k in ks:
        k_use = min(int(k), n)
        out[f"Recall@{k}"] = float(np.mean(np.any(ranks[:, :k_use] == targets, axis=1)))
    return out

def find_task2_paths():
    for sr in [
        "/kaggle/input/competitions/task-2-datavidia-11-0",
        "/kaggle/input/task-2-datavidia-11-0",
        "/kaggle/input",
        ".",
    ]:
        if not os.path.exists(sr):
            continue
        for root, _, files in os.walk(sr):
            if "sample_submission.csv" in files:
                cand_sub = os.path.join(root, "sample_submission.csv")
                probe_id = str(pd.read_csv(cand_sub, nrows=1)["Id"].iloc[0])
                if os.path.exists(os.path.join(root, probe_id, "pre.jpg")):
                    return root, cand_sub
                if os.path.exists(os.path.join(root, "Fenomena", probe_id, "pre.jpg")):
                    return os.path.join(root, "Fenomena"), cand_sub
    raise FileNotFoundError("Dataset Task-2 (sample_submission.csv + <Id>/pre.jpg) tidak ditemukan.")


def build_pairs_df(data_dir: str, sub_path: str) -> pd.DataFrame:
    """Bangun DataFrame pasangan pre/post sesuai persis urutan Id di sample_submission.csv."""
    all_ids = pd.read_csv(sub_path)["Id"].astype(str).tolist()
    return pd.DataFrame({
        "Id": all_ids,
        "pre_path": [os.path.join(data_dir, uid, "pre.jpg") for uid in all_ids],
        "post_path": [os.path.join(data_dir, uid, "post.jpg") for uid in all_ids],
    })


def extract_embeddings_for_seno(
    pairs_df: pd.DataFrame,
    weights_dir: str = None,
    backbone_choice: str = BACKBONE_CHOICE,
    include_gray: bool = False,
    batch_size: int = BATCH_SIZE,
):
    """Fungsi terkunci untuk 01_model.ipynb & 02_submission.ipynb (offline load dari weights/)."""
    set_reproducible_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, b_info = load_dino_backbone(
        backbone_choice=backbone_choice,
        device=device,
        local_weights_dir=weights_dir,
        save_weights_dir=None,
    )
    if ALLOW_DATAPARALLEL and device == "cuda" and torch.cuda.device_count() > 1:
        backbone = nn.DataParallel(backbone)

    c_pre, c_post, _, _ = run_pairwise_extraction(
        backbone, pairs_df, b_info, mode="color_norm", compute_patch_stats=False, device=device, batch_size=batch_size
    )
    out = {
        "Id": pairs_df["Id"].astype(str).tolist(),
        "dino_color_pre": c_pre,
        "dino_color_post": c_post,
        "meta": b_info,
    }
    if include_gray:
        g_pre, g_post, _, _ = run_pairwise_extraction(
            backbone, pairs_df, b_info, mode="gray3", compute_patch_stats=False, device=device, batch_size=batch_size
        )
        out["dino_gray_pre"] = g_pre
        out["dino_gray_post"] = g_post
    return out
