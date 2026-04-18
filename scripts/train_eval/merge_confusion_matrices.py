#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # Add project root to sys.path

import glob
import argparse
from PIL import Image, ImageDraw, ImageFont

from modules.config import RESULTS_LIGHT_PATH

def find_subset_dirs(root, pattern):
    dirs = sorted(glob.glob(os.path.join(root, pattern)))
    return [d for d in dirs if os.path.isdir(d)]

def detect_models(first_subset_dir):
    # model folders are immediate children (e.g. occft_convnext)
    names = sorted([n for n in os.listdir(first_subset_dir)
                    if os.path.isdir(os.path.join(first_subset_dir, n))])
    return names

def subset_display_name(subset_dir):
    b = os.path.basename(subset_dir)
    if 'models_' in b and '-testset' in b:
        return b.split('models_')[-1].split('-testset')[0]
    return b

def find_cm_image(subset_dir, model_name, mode):
    cm_dir = os.path.join(subset_dir, model_name, 'confusion_matrix')
    if not os.path.isdir(cm_dir):
        return None
    if mode == 'normalized':
        matches = glob.glob(os.path.join(cm_dir, '*_cm_normalized.*'))
    else:
        matches = glob.glob(os.path.join(cm_dir, '*_cm.*'))
        matches = [m for m in matches if not m.lower().endswith('_cm_normalized.png')]
    return matches[0] if matches else None

def load_and_normalize_images(paths, target_size=None, bg=(255,255,255)):
    imgs = []
    for p in paths:
        im = Image.open(p).convert('RGBA')
        if target_size:
            im.thumbnail(target_size, Image.LANCZOS)
            bg_im = Image.new('RGBA', target_size, bg + (255,))
            x = (target_size[0] - im.width)//2
            y = (target_size[1] - im.height)//2
            bg_im.paste(im, (x,y), im)
            imgs.append(bg_im.convert('RGB'))
        else:
            imgs.append(im.convert('RGB'))
    return imgs

def _text_size(draw, text, font):
    # Robust text size: prefer draw.textbbox, fall back to font.getsize
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            return (len(text) * 8, getattr(font, "size", 12))

def make_compound_image(images, titles, model_name, out_path,
                        per_row=4, cell_size=None, padding=10, title_height=36):
    cols = per_row
    rows = (len(images) + cols - 1)//cols
    cell_w, cell_h = cell_size if cell_size else (max(i.width for i in images), max(i.height for i in images))
    title_space = 20
    top_title_space = title_height + 10

    canvas_w = cols*cell_w + (cols+1)*padding
    canvas_h = top_title_space + rows*(cell_h + title_space) + (rows+1)*padding

    canvas = Image.new('RGB', (canvas_w, canvas_h), (255,255,255))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    top_title = f"Confusion matrices for {model_name} on different subsets of the occluded test set"
    w_title, h_title = _text_size(draw, top_title, title_font)
    draw.text(((canvas_w-w_title)//2, 6), top_title, fill=(0,0,0), font=title_font)

    for idx, (img, t) in enumerate(zip(images, titles)):
        r = idx // cols
        c = idx % cols
        x = padding + c*(cell_w + padding)
        y = top_title_space + padding + r*(cell_h + title_space + padding)

        w_sub, h_sub = _text_size(draw, t, small_font)
        draw.text((x + (cell_w-w_sub)//2, y), t, fill=(0,0,0), font=small_font)
        img_y = y + h_sub + 4
        canvas.paste(img.resize((cell_w, cell_h)), (x, img_y))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, quality=95)
    print("Saved", out_path)

def process(root, pattern, output_dir, mode, per_row=4, target_cell=(360,360)):
    subset_dirs = find_subset_dirs(root, pattern)
    if not subset_dirs:
        raise SystemExit("No subset dirs found with that pattern.")

    models = detect_models(subset_dirs[0])
    if not models:
        raise SystemExit("No model directories found in first subset: " + subset_dirs[0])

    for model in models:
        img_paths = []
        titles = []
        for sd in subset_dirs:
            p = find_cm_image(sd, model, mode)
            titles.append(subset_display_name(sd))
            img_paths.append(p if p else None)

        processed_paths = []
        for p in img_paths:
            if p and os.path.isfile(p):
                processed_paths.append(p)
            else:
                placeholder = Image.new('RGB', target_cell, (240,240,240))
                draw = ImageDraw.Draw(placeholder)
                try:
                    f = ImageFont.truetype("arial.ttf", 12)
                except:
                    f = ImageFont.load_default()
                draw.text((10,10), "Missing", fill=(0,0,0), font=f)
                tmp_path = os.path.join(output_dir, f"_tmp_missing_{os.getpid()}_{len(processed_paths)}.png")
                os.makedirs(output_dir, exist_ok=True)
                placeholder.save(tmp_path)
                processed_paths.append(tmp_path)

        imgs = load_and_normalize_images(processed_paths, target_size=target_cell)
        if mode == 'normalized':
            out_name = f"normalized_confusion_matrices_{model}.png"
        else:
            out_name = f"confusion_matrices_{model}.png"
        out_path = os.path.join(output_dir, out_name)
        make_compound_image(imgs, titles, model, out_path, per_row=per_row, cell_size=target_cell)

    for f in glob.glob(os.path.join(output_dir, "_tmp_missing_*.png")):
        try:
            os.remove(f)
        except: pass

# >>> Run with more resolution and normalized CMs:
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/merge_confusion_matrices.py" --cell-size 600 600
# >>> Run with more resolution and raw CMs:
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/merge_confusion_matrices.py" --cell-size 600 600 --mode raw
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="20260401-*_cmplt-run_occft-models_*-testset_do-evaluation-completely-keras",
                   help="glob pattern to find subset directories")
    p.add_argument("--output", default=None, help="output directory (overridden to RESULTS_LIGHT_PATH/xai_subsets if not set)")
    p.add_argument("--mode", choices=("normalized","raw"), default="normalized",
                   help="which CM to collect: 'normalized' searches for *_cm_normalized.*; 'raw' searches for *_cm.* excluding normalized")
    p.add_argument("--per-row", type=int, default=4)
    p.add_argument("--cell-size", type=int, nargs=2, metavar=('W','H'), default=(360,360))
    args = p.parse_args()

    hard_root = os.path.join(RESULTS_LIGHT_PATH, "xai_subsets", "confusion_matrices")
    hard_output = os.path.join(RESULTS_LIGHT_PATH, "xai_subsets", "confusion_matrices")
    process(hard_root, args.pattern, hard_output, mode=args.mode, per_row=args.per_row, target_cell=tuple(args.cell_size))