"""Split BW true_pet.npz into per-subject files under GitHub's 100 MB limit."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/MIAProject/SAM_KEM_QWEN3_V3_REPO")
BW_RAW = Path(r"D:/MIAProject/brainweb/数据/brainweb_petmr")
SPLIT_V3 = Path(r"D:/MIAProject/SAM_KEM_QWEN3_MEDSAM2_DELIVERY/splits/subject_split.csv")
OUT = ROOT / "data" / "raw_pet"
OUT.mkdir(parents=True, exist_ok=True)

# Load the existing combined BW NPZ (or extract fresh)
combined_p = OUT / "raw_pet_brainweb.npz"
print(f"Loading {combined_p}...")
z = np.load(combined_p)

# Convert float32 -> float16 to halve the size
print("Converting to float16...")
new_data = {}
for k, v in z.items():
    if v.dtype == np.float32:
        new_data[k] = v.astype(np.float16)
    else:
        new_data[k] = v

# Save per subject (one file per subject to keep each < 100 MB)
# Each (256, 256, 258) float16 = ~33 MB. One file per subject is fine.
sub_keys = {}
for k in new_data:
    if "__true_pet" in k:
        sid = k.split("__")[0]
        sub_keys[sid] = True

manifest = {"brainweb": {"subjects": {}}, "hecktor": None}
for sid in sorted(sub_keys):
    case = {f"{sid}__true_pet": new_data[f"{sid}__true_pet"],
            f"{sid}__slice_count": new_data[f"{sid}__slice_count"]}
    sub_p = OUT / f"raw_pet_brainweb_{sid}.npz"
    np.savez_compressed(sub_p, **case)
    sz_mb = sub_p.stat().st_size / 1e6
    print(f"  {sid}: {sz_mb:.1f} MB")
    manifest["brainweb"]["subjects"][sid] = {
        "file": sub_p.name,
        "size_mb": round(sz_mb, 1),
        "shape": "256x256x258 float16",
    }

# Keep the combined file too but smaller (already saved). Actually remove it
# since we have per-subject files now.
if combined_p.exists():
    combined_p.unlink()
    print(f"removed {combined_p}")

# Update manifest
manifest["brainweb"]["n_subjects"] = len(sub_keys)
manifest["brainweb"]["format"] = "One .npz per subject (float16, ~33 MB each)"
manifest["brainweb"]["source"] = "D:/MIAProject/brainweb/数据/brainweb_petmr/<sid>/sim_0/true_pet.nii.gz"

# Read existing HE info and merge
manifest_p = OUT / "RAW_PET_MANIFEST.json"
if manifest_p.exists():
    existing = json.load(open(manifest_p, encoding="utf-8"))
    if "hecktor" in existing:
        manifest["hecktor"] = existing["hecktor"]

with open(manifest_p, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"\nwrote {manifest_p}")
print(f"\nBW files: {len(sub_keys)} × ~33 MB = ~{len(sub_keys)*33} MB total")