#!/usr/bin/env python3
"""Phase 2: generate T2V videos for TAP-Bench probe cells on a cloud GPU.

Reads gt_curves.csv (prompts + GT parameters), generates one video per unique
probe cell with the selected model, exports mp4 + a generation manifest.

Models (all fit a 24GB GPU):
  cogvideox2b  THUDM/CogVideoX-2b          49f @ 8 fps
  ltxvideo     Lightricks/LTX-Video        97f @ 25 fps
  wan13b       Wan-AI/Wan2.1-T2V-1.3B-Diffusers  81f @ 16 fps

Subsets:
  full   every unique cell, all seeds
  curve  wheel seeds 1-2 (all cells) + wheel f_tex>=12 seeds 3-5 + grating seed 1
  mini   wheel seed 1 (all cells) + grating seed 1
  i2vset wheel seed 1, aliasing cells (f_tex>=12) + slow controls (f_tex<=4)

Modes:
  t2v  text-to-video (default)
  i2v  first-frame conditioned (ltxvideo only; deconfounds prompt following)

Usage:
  python generate_probe.py --model cogvideox2b --subset curve --gt gt_curves.csv
  python generate_probe.py --model ltxvideo --mode i2v --subset i2vset --gt gt_curves.csv
"""

import argparse
import csv
import hashlib
import os

CELL_STEM = lambda clip_id: "_".join(clip_id.split("/")[1].split("_")[:-1])


def parse_cell(stem, category):
    if category == "wheel":
        parts = stem.split("_")
        return {"k": int(parts[0][1:]), "omega": float(parts[1][1:]),
                "seed": int(parts[2][1:])}
    if category == "grating":
        parts = stem.split("_")
        return {"v": float(parts[0][1:]), "fs": int(parts[1][1:]) / 1000.0,
                "seed": int(parts[2][1:])}
    return {"seed": 0}


def f_tex_of(category, p):
    if category == "wheel":
        return p["k"] * p["omega"]
    if category == "grating":
        return p["v"] * p["fs"]
    return 0.0


def load_cells(gt_csv):
    cells = {}
    with open(gt_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row["category"]
            if cat == "ball":
                continue
            stem = CELL_STEM(row["clip_id"])
            if stem not in cells:
                cells[stem] = {"category": cat, "stem": stem,
                               "prompt": row["prompt"],
                               **parse_cell(stem, cat)}
    return cells


def select_subset(cells, subset):
    out = []
    for c in cells.values():
        s, ft = c["seed"], f_tex_of(c["category"], c)
        if subset == "full":
            take = True
        elif subset == "mini":
            take = s == 1
        elif subset == "i2vset":
            take = (c["category"] == "wheel" and s == 1
                    and (ft >= 12 or ft <= 4))
        else:
            take = (c["category"] == "grating" and s == 1) or \
                   (c["category"] == "wheel" and s in (1, 2)) or \
                   (c["category"] == "wheel" and s in (3, 4, 5) and ft >= 12)
        if take:
            out.append(c)
    out.sort(key=lambda c: (c["category"], c["stem"]))
    return out


MODEL_SPECS = {
    "cogvideox2b": dict(
        repo="THUDM/CogVideoX-2b", fps=8, width=720, height=480,
        num_frames=49, steps=50, guidance=6.0, dtype="float16",
        negative=""),
    "ltxvideo": dict(
        repo="Lightricks/LTX-Video", fps=25, width=768, height=512,
        num_frames=97, steps=40, guidance=3.0, dtype="bfloat16",
        negative="worst quality, inconsistent motion, blurry, jittery, distorted"),
    "wan13b": dict(
        repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", fps=16, width=832, height=480,
        num_frames=81, steps=50, guidance=5.0, dtype="bfloat16",
        negative=("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
                  "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
                  "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
                  "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")),
}

DTYPE = {"float16": "float16", "bfloat16": "bfloat16"}


def build_pipe(name, spec, mode, offload):
    import torch
    from diffusers import (CogVideoXPipeline, LTXImageToVideoPipeline,
                           LTXPipeline, WanPipeline)
    dt = getattr(torch, DTYPE[spec["dtype"]])
    if name == "cogvideox2b":
        pipe = CogVideoXPipeline.from_pretrained(spec["repo"], torch_dtype=dt)
        pipe.vae.enable_tiling()
    elif name == "ltxvideo":
        cls = LTXImageToVideoPipeline if mode == "i2v" else LTXPipeline
        pipe = cls.from_pretrained(spec["repo"], torch_dtype=dt)
        pipe.vae.enable_tiling()
    else:
        if mode == "i2v":
            raise SystemExit("i2v only supported for ltxvideo")
        pipe = WanPipeline.from_pretrained(spec["repo"], torch_dtype=dt)
        pipe.vae.enable_tiling()
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def cell_seed(stem):
    h = hashlib.sha256(stem.encode()).digest()
    return int.from_bytes(h[:4], "little")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    ap.add_argument("--subset", default="curve",
                    choices=["full", "curve", "mini", "i2vset"])
    ap.add_argument("--mode", default="t2v", choices=["t2v", "i2v"])
    ap.add_argument("--gt", default="gt_curves.csv")
    ap.add_argument("--out", default="gen_out")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--lora", default=None,
                    help="path to LoRA weights to load before generation")
    ap.add_argument("--prompt-strip", default=None,
                    help="substring removed from every prompt")
    ap.add_argument("--prompt-append", default=None,
                    help="string appended to every prompt")
    args = ap.parse_args()

    import torch
    from diffusers.utils import export_to_video

    spec = MODEL_SPECS[args.model]
    cells = select_subset(load_cells(args.gt), args.subset)
    print(f"{len(cells)} cells for model={args.model} mode={args.mode} "
          f"subset={args.subset}", flush=True)
    out_dir = os.path.join(args.out, args.model, args.mode)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out, f"{args.model}_{args.mode}_manifest.csv")
    done = set()
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if os.path.isfile(r["path"]) and os.path.getsize(r["path"]) > 0:
                    done.add(r["cell_key"])

    pipe = None
    new_rows = []
    for i, c in enumerate(cells):
        for s_i in range(args.samples):
            key = f"{c['stem']}_rep{s_i}"
            if key in done:
                continue
            prompt = c["prompt"]
            if args.prompt_strip and args.prompt_strip in prompt:
                prompt = prompt.replace(args.prompt_strip, "")
            if args.prompt_append:
                prompt = prompt + args.prompt_append
            if pipe is None:
                print("loading pipeline...", flush=True)
                pipe = build_pipe(args.model, spec, args.mode, args.offload)
                if args.lora:
                    pipe.load_lora_weights(args.lora)
                    print(f"loaded LoRA: {args.lora}", flush=True)
            path = os.path.join(out_dir, f"{key}.mp4")
            gen = torch.Generator(device="cuda").manual_seed(cell_seed(key))
            kw = dict(prompt=prompt, negative_prompt=spec["negative"],
                      width=spec["width"], height=spec["height"],
                      num_frames=spec["num_frames"],
                      num_inference_steps=spec["steps"],
                      guidance_scale=spec["guidance"], generator=gen)
            if args.mode == "i2v":
                from PIL import Image
                cond = os.path.join("first_frames", c["category"],
                                    c["stem"] + ".png")
                img = Image.open(cond).convert("RGB")
                img = img.resize((spec["width"], spec["height"]))
                kw["image"] = img
            video = pipe(**kw).frames[0]
            export_to_video(video, path, fps=spec["fps"])
            new_rows.append({
                "model": args.model, "mode": args.mode, "cell_key": key,
                "category": c["category"], "stem": c["stem"],
                "k": c.get("k", ""), "omega_gt": c.get("omega", ""),
                "v_px_s": c.get("v", ""), "fs_cyc_px": c.get("fs", ""),
                "f_tex_hz": round(f_tex_of(c["category"], c), 3),
                "fps_model": spec["fps"], "path": path,
            })
            print(f"[{i + 1}/{len(cells)}] {key} -> {path}", flush=True)
            with open(manifest_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(new_rows[-1]))
                if f.tell() == 0:
                    w.writeheader()
                w.writerows(new_rows[-1:])
    print(f"done; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
