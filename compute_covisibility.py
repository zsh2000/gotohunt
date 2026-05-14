"""
Pre-compute per-scene MegaLoc cosine-similarity ("covisibility") matrices for
any of the datasets supported by omni_evaluation_code.py.

For each scene, this script:
  1. Enumerates the RGB frames in the dataset's image directory.
  2. Extracts MegaLoc global descriptors (322x322, ImageNet-normalized).
  3. Saves the NxN cosine-similarity matrix as `similarity_matrix.npy`
     under <output_root>/<scene_name>/, matching the layout expected by
     `--covisibility_root` in omni_evaluation_code.py.

Pass the resulting `<output_root>` to omni_evaluation_code.py:

    --frame_strategy diverse --covisibility_root <output_root>

Usage (Bonn):
    python compute_covisibility.py \\
        --dataset bonn \\
        --data_root data/eval/bonn/rgbd_bonn_dataset \\
        --output_root /path/to/covisibility/bonn

Usage (7-Scenes):
    python compute_covisibility.py \\
        --dataset 7scenes \\
        --data_root data/eval/7scenes \\
        --output_root /path/to/covisibility/7scenes
"""

import argparse
import glob
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ── Dataset → image-folder resolver ───────────────────────────────────────────
# Mirrors omni_evaluation_code.discover_scenes so scene names match exactly.

SINTEL_SEQ_LIST = [
    "alley_2", "ambush_4", "ambush_5", "ambush_6",
    "cave_2", "cave_4", "market_2", "market_5", "market_6",
    "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3",
]

BONN_SEQ_LIST = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]


def _existing(*candidates):
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def list_scenes(dataset, data_root):
    """Return [(scene_name, image_dir), ...] for the chosen dataset.

    scene_name must match what omni_evaluation_code.discover_scenes produces.
    image_dir is the folder of RGB frames to descriptor-encode.
    """
    if dataset == "nrgbd":
        scenes = sorted(d for d in os.listdir(data_root)
                        if os.path.isdir(os.path.join(data_root, d)))
        return [(s, os.path.join(data_root, s, "images")) for s in scenes]

    if dataset == "7scenes":
        result = []
        for scene in sorted(os.listdir(data_root)):
            scene_path = os.path.join(data_root, scene)
            if not os.path.isdir(scene_path):
                continue
            for seq in sorted(os.listdir(scene_path)):
                seq_path = os.path.join(scene_path, seq)
                if seq.startswith("seq-") and os.path.isdir(seq_path):
                    result.append((f"{scene}/{seq}", seq_path))
        return result

    if dataset in ("tum", "tum_full", "tum_mast3r"):
        result = []
        for d in sorted(os.listdir(data_root)):
            scene_path = os.path.join(data_root, d)
            if not os.path.isdir(scene_path):
                continue
            rgb = _existing(
                os.path.join(scene_path, "rgb_90"),
                os.path.join(scene_path, "rgb_sync_for_pose_5_90"),
                os.path.join(scene_path, "rgb"),
            )
            if rgb is not None:
                result.append((d, rgb))
        return result

    if dataset == "bonn":
        return [
            (s, os.path.join(data_root, f"rgbd_bonn_{s}", "rgb"))
            for s in BONN_SEQ_LIST
        ]

    if dataset == "bonn_full":
        result = []
        for d in sorted(os.listdir(data_root)):
            if not d.startswith("rgbd_bonn_"):
                continue
            scene_path = os.path.join(data_root, d)
            rgb = _existing(os.path.join(scene_path, "rgb"))
            if rgb is not None:
                result.append((d[len("rgbd_bonn_"):], rgb))
        return result

    if dataset == "kitti":
        img_root = os.path.join(data_root, "image_gathered")
        return [
            (s, os.path.join(img_root, s))
            for s in sorted(os.listdir(img_root))
            if os.path.isdir(os.path.join(img_root, s))
        ]

    if dataset == "sintel":
        return [
            (s, os.path.join(data_root, "training", "final", s))
            for s in SINTEL_SEQ_LIST
        ]

    if dataset in ("scannet", "scannet_full"):
        result = []
        for d in sorted(os.listdir(data_root)):
            scene_path = os.path.join(data_root, d)
            rgb = _existing(
                os.path.join(scene_path, "ordered_color"),
                os.path.join(scene_path, "color"),
            )
            if rgb is not None:
                result.append((d, rgb))
        return result

    raise ValueError(f"Unsupported dataset: {dataset}")


# ── MegaLoc backbone ──────────────────────────────────────────────────────────

def load_megaloc(megaloc_repo=None, megaloc_weights=None):
    """Load MegaLoc, either from torch.hub or from a local checkout + safetensors."""
    if megaloc_repo and megaloc_weights:
        sys.path.insert(0, megaloc_repo)
        from megaloc_model import MegaLoc
        from safetensors.torch import load_file

        model = MegaLoc()
        state = load_file(megaloc_weights)
        model.load_state_dict(state)
    else:
        # Fallback: pull via torch.hub (requires internet on first run)
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model", trust_repo=True)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


# ── Per-scene pipeline ────────────────────────────────────────────────────────

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg")


def get_image_paths(image_dir):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(paths)


def extract_descriptors(model, image_paths, batch_size):
    tfm = transforms.Compose([
        transforms.Resize((322, 322)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    device = next(model.parameters()).device
    descriptors = []
    t_load = t_fwd = 0.0

    for i in tqdm(range(0, len(image_paths), batch_size), desc="  descriptors"):
        batch_paths = image_paths[i : i + batch_size]

        t0 = time.perf_counter()
        batch = torch.stack([tfm(Image.open(p).convert("RGB"))
                             for p in batch_paths]).to(device)
        t_load += time.perf_counter() - t0

        t1 = time.perf_counter()
        with torch.no_grad():
            feat = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd += time.perf_counter() - t1

        descriptors.append(feat.cpu())

    n = len(image_paths)
    print(f"    load={t_load:.1f}s ({t_load/n*1000:.1f}ms/img), "
          f"forward={t_fwd:.1f}s ({t_fwd/n*1000:.1f}ms/img)")
    return torch.cat(descriptors, dim=0)


def save_similarity_plot(sim, out_dir, scene_name):
    n = sim.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"{scene_name} — similarity ({n}×{n})")
    ax.set_xlabel("frame index"); ax.set_ylabel("frame index")
    plt.colorbar(im, ax=ax, label="cosine similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "similarity_matrix.png"), dpi=150)
    plt.close()


def process_scene(model, image_dir, output_dir, scene_name, batch_size,
                  save_descriptors=False, save_plot=True):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = get_image_paths(image_dir)
    if not image_paths:
        print(f"  [SKIP] no images in {image_dir}")
        return False

    print(f"  {len(image_paths)} frames in {image_dir}")
    descriptors = extract_descriptors(model, image_paths, batch_size)
    sim = (descriptors @ descriptors.T).numpy()

    np.save(os.path.join(output_dir, "similarity_matrix.npy"), sim)
    if save_descriptors:
        np.save(os.path.join(output_dir, "descriptors.npy"), descriptors.numpy())
    with open(os.path.join(output_dir, "frame_names.txt"), "w") as f:
        f.writelines(os.path.basename(p) + "\n" for p in image_paths)
    if save_plot:
        save_similarity_plot(sim, output_dir, scene_name)

    off = sim[~np.eye(sim.shape[0], dtype=bool)]
    print(f"    saved {sim.shape}, off-diag min={off.min():.3f} "
          f"max={off.max():.3f} mean={off.mean():.3f}")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   choices=["7scenes", "nrgbd", "tum", "tum_full", "tum_mast3r",
                            "bonn", "bonn_full", "kitti", "sintel",
                            "scannet", "scannet_full"])
    p.add_argument("--data_root", required=True,
                   help="Same path passed to omni_evaluation_code.py --data_root.")
    p.add_argument("--output_root", required=True,
                   help="Where per-scene similarity_matrix.npy files are written. "
                        "Pass this to omni_evaluation_code.py --covisibility_root.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--megaloc_repo", default=None,
                   help="Local clone of gmberton/MegaLoc (skip to use torch.hub).")
    p.add_argument("--megaloc_weights", default=None,
                   help="Path to MegaLoc model.safetensors (used with --megaloc_repo).")
    p.add_argument("--save_descriptors", action="store_true",
                   help="Also dump per-scene descriptors.npy.")
    p.add_argument("--no_plot", action="store_true",
                   help="Skip the similarity_matrix.png visualization.")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute scenes whose similarity_matrix.npy already exists.")
    p.add_argument("--scenes", nargs="+", default=None,
                   help="Restrict to these scene names (as listed by --dry_run).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the discovered scene list and exit.")
    args = p.parse_args()

    scenes = list_scenes(args.dataset, args.data_root)
    if args.scenes is not None:
        scenes = [(n, d) for n, d in scenes if n in args.scenes]
    print(f"Discovered {len(scenes)} scene(s).")

    if args.dry_run:
        for n, d in scenes:
            print(f"  {n}  ->  {d}")
        return

    os.makedirs(args.output_root, exist_ok=True)
    print("Loading MegaLoc ...")
    model = load_megaloc(args.megaloc_repo, args.megaloc_weights)

    for i, (scene_name, image_dir) in enumerate(scenes):
        out_dir = os.path.join(args.output_root, scene_name)
        done = os.path.join(out_dir, "similarity_matrix.npy")
        if os.path.exists(done) and not args.overwrite:
            print(f"[{i+1}/{len(scenes)}] {scene_name}  (skipped — already exists)")
            continue
        print(f"[{i+1}/{len(scenes)}] {scene_name}")
        process_scene(
            model, image_dir, out_dir, scene_name,
            batch_size=args.batch_size,
            save_descriptors=args.save_descriptors,
            save_plot=not args.no_plot,
        )
        print()

    print(f"Done. Outputs under {args.output_root}/")


if __name__ == "__main__":
    main()
