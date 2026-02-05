import os
import re
import numpy as np
import nibabel as nib
from PIL import Image

PRED_DIR = r"FSbase_model_output_Synapse/re599predictions"
MAP_TXT = r"./case_slice_map.txt"
OUT_DIR = r"re599predictions"
NUM_CLASSES = 9
# ===============================

COLORMAP_RGB = np.array([
    [0, 0, 0],  # 0
    [51, 68, 161],  # 1
    [116, 204, 77],  # 2
    [221, 46, 33],  # 3
    [157, 227, 221],  # 4
    [182, 70, 174],  # 5
    [235, 227, 52],  # 6
    [106, 197, 225],  # 7
    [242, 240, 233],  # 8
], dtype=np.uint8)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def normalize_to_uint8(img2d: np.ndarray) -> np.ndarray:
    x = np.array(img2d, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - mn) / (mx - mn)
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    return x


def get_depth_axis(vol3d: np.ndarray) -> int:
    return int(np.argmin(vol3d.shape))


def slice2d(vol3d: np.ndarray, s: int) -> np.ndarray:
    v = np.asarray(vol3d)
    if v.ndim != 3:
        raise ValueError(f"volume dim != 3, got {v.shape}")
    d_axis = get_depth_axis(v)
    D = v.shape[d_axis]
    s = int(np.clip(s, 0, D - 1))
    if d_axis == 0:  # (D,H,W)
        return v[s, :, :]
    elif d_axis == 2:  # (H,W,D)
        return v[:, :, s]
    else:  # (H,D,W)
        return v[:, s, :]


def mask_to_uint8(mask2d: np.ndarray) -> np.ndarray:
    m = np.rint(mask2d).astype(np.int32)
    m = np.clip(m, 0, NUM_CLASSES - 1).astype(np.uint8)
    return m


def mask_to_color(mask2d: np.ndarray) -> np.ndarray:
    m = mask_to_uint8(mask2d).astype(np.int32)
    return COLORMAP_RGB[m]  # (H,W,3)


def overlay_direct(gray_u8: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    img_rgb = np.repeat(gray_u8[:, :, None], 3, axis=2).astype(np.uint8)
    color = mask_to_color(mask2d).astype(np.uint8)
    m = mask_to_uint8(mask2d).astype(np.int32)
    fg = (m > 0)
    out = img_rgb.copy()
    out[fg] = color[fg]
    return out


def parse_map(txt_path: str):
    mp = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line, maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"bad line: {line}")
            case_id = parts[0].strip()
            s_part = parts[1].replace(" ", "")
            s_list = [int(x) for x in s_part.split(",") if x != ""]
            mp[case_id] = s_list
    return mp


def list_available_cases(pred_dir: str):
    cases = set()
    for fn in os.listdir(pred_dir):
        if fn.endswith("_img.nii.gz"):
            cases.add(fn.replace("_img.nii.gz", ""))
    return sorted(cases)


def resolve_case_id(case_id: str, available_cases):
    if case_id in available_cases:
        return case_id
    m = re.search(r"(\d+)$", case_id)
    if m:
        n = int(m.group(1))
        cand = f"case{n:04d}"
        if cand in available_cases:
            return cand
    return case_id


def main():
    ensure_dir(OUT_DIR)
    mp = parse_map(MAP_TXT)
    available = list_available_cases(PRED_DIR)

    for raw_case, slices in mp.items():
        case = resolve_case_id(raw_case, available)
        img_p = os.path.join(PRED_DIR, f"{case}_img.nii.gz")
        gt_p = os.path.join(PRED_DIR, f"{case}_gt.nii.gz")
        pred_p = os.path.join(PRED_DIR, f"{case}_pred.nii.gz")

        if not (os.path.exists(img_p) and os.path.exists(gt_p) and os.path.exists(pred_p)):
            print(f"[skip] {raw_case} -> {case} 缺文件：")
            print(" ", img_p, os.path.exists(img_p))
            print(" ", gt_p, os.path.exists(gt_p))
            print(" ", pred_p, os.path.exists(pred_p))
            continue

        img_v = nib.load(img_p).get_fdata()
        gt_v = nib.load(gt_p).get_fdata()
        pred_v = nib.load(pred_p).get_fdata()

        out_case_dir = os.path.join(OUT_DIR, case)
        ensure_dir(out_case_dir)

        for s in slices:
            img2d = slice2d(img_v, s)
            gt2d = slice2d(gt_v, s)
            pred2d = slice2d(pred_v, s)

            gray = normalize_to_uint8(img2d)
            gt_overlay = overlay_direct(gray, gt2d)
            pred_overlay = overlay_direct(gray, pred2d)
            pred_color_mask = mask_to_color(pred2d).astype(np.uint8)

            gt_color_mask = mask_to_color(gt2d).astype(np.uint8)

            Image.fromarray(gray).save(os.path.join(out_case_dir, f"{case}_slice{s:03d}_img.png"))
            Image.fromarray(gt_overlay).save(os.path.join(out_case_dir, f"{case}_slice{s:03d}_gt_overlay.png"))
            Image.fromarray(pred_overlay).save(os.path.join(out_case_dir, f"{case}_slice{s:03d}_pred_overlay.png"))

            Image.fromarray(pred_color_mask, mode="RGB").save(
                os.path.join(out_case_dir, f"{case}_slice{s:03d}_pred_mask_color.png")
            )
            Image.fromarray(gt_color_mask, mode="RGB").save(
                os.path.join(out_case_dir, f"{case}_slice{s:03d}_gt_mask_color.png")
            )


if __name__ == "__main__":
    main()
