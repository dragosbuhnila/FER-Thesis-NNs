import os, sys, math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
import re

from modules.config import (
    BUBBLES_OCC_OCC_DIR_PATH,
    CONFUSION_OCC_OCC_MATRICES_DIR_PATH,
    EXTERNAL_OCC_OCC_DIR_PATH,
    GRADCAM_OCC_OCC_DIR_PATH,
    MERGED_OCC_OCC_HEATMAPS_PDFS_DIR_PATH,
)

# Layout settings (pixels) — much narrower model-name column to save space
ROWS = 7            # up to 7 models per page (one row per model)
COLS = 4            # CONFUSION_MATRICES, Bubbles, EXTERNAL, GRADCAM
MODEL_NAME_W = 48   # << reduced to save horizontal space
CELL_W = 360
CELL_H = 240
TOP_MARGIN_H = 18
H_SPACING = 10
V_SPACING = 10
PLACEHOLDER_BG = (240, 240, 240)
PLACEHOLDER_TEXT = "MISSING"

# Border settings
BORDER_COLOR = (0, 0, 0)
BORDER_WIDTH = 2

# Shrink factor for confusion matrices (first column)
CELL_SCALE_CM = 0.88

def natural_sort_key(s):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def find_last_gradcam_layer(model_folder):
    model_path = os.path.join(GRADCAM_OCC_OCC_DIR_PATH, model_folder)
    if not os.path.isdir(model_path):
        return None
    subdirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
    if not subdirs:
        return None
    if "occft_inceptionv3" in model_folder.lower():
        for d in subdirs:
            if "layer_30" in d:
                return d
    for d in subdirs:
        if "last" in d.lower():
            return d
    best = None
    best_num = -1
    for d in subdirs:
        m = re.search(r'layer_(\d+)', d)
        if m:
            n = int(m.group(1))
            if n > best_num:
                best_num = n
                best = d
    if best:
        return best
    subdirs.sort(key=natural_sort_key)
    return subdirs[-1]

def find_csm_for_model(model_folder):
    """Return list of 4 image paths: [CONFUSION_MATRICES, Bubbles, EXTERNAL, GRADCAM] or None."""
    res = []
    cm_path = os.path.join(CONFUSION_OCC_OCC_MATRICES_DIR_PATH, model_folder, f"{model_folder}_cm.png")
    res.append(cm_path if os.path.exists(cm_path) else None)
    b1 = os.path.join(BUBBLES_OCC_OCC_DIR_PATH, model_folder, "CSM_sottrazione_norm.png")
    b2 = os.path.join(BUBBLES_OCC_OCC_DIR_PATH, model_folder, "CSM.png")
    res.append(b1 if os.path.exists(b1) else (b2 if os.path.exists(b2) else None))
    ex_path = os.path.join(EXTERNAL_OCC_OCC_DIR_PATH, model_folder, "CSM.png")
    res.append(ex_path if os.path.exists(ex_path) else None)
    layer = find_last_gradcam_layer(model_folder)
    if layer:
        g_path = os.path.join(GRADCAM_OCC_OCC_DIR_PATH, model_folder, layer, "CSM.png")
        res.append(g_path if os.path.exists(g_path) else None)
    else:
        res.append(None)
    return res

def get_text_size(draw, text, font):
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0,0), text, font=font)
        return (bbox[2]-bbox[0], bbox[3]-bbox[1])
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return (bbox[2]-bbox[0], bbox[3]-bbox[1])
    if hasattr(font, "getsize"):
        return font.getsize(text)
    try:
        return draw.textsize(text, font=font)
    except Exception:
        return (len(text)*6, 10)

def make_placeholder(text, w, h):
    img = Image.new("RGB", (w, h), color=PLACEHOLDER_BG)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    tw, th = get_text_size(draw, text, font)
    draw.text(((w-tw)/2, (h-th)/2), text, fill=(100,100,100), font=font)
    return img

def load_and_fit(path, max_w, max_h):
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return make_placeholder(PLACEHOLDER_TEXT, max_w, max_h)
    iw, ih = img.size
    if iw == 0 or ih == 0:
        return make_placeholder(PLACEHOLDER_TEXT, max_w, max_h)
    ratio = min(max_w/iw, max_h/ih)
    new_w = max(1, int(iw*ratio))
    new_h = max(1, int(ih*ratio))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return img

def create_stacked_vertical_text_image(text, max_width, max_height):
    """
    Create a narrow RGBA image with the model name rendered as stacked characters.
    Chooses a TrueType font if available and scales it so the stacked text fits.
    """
    # prefer a commonly-available truetype font
    try:
        font_path = "DejaVuSans.ttf"
        base_size = 20
        font = ImageFont.truetype(font_path, base_size)
    except Exception:
        font = ImageFont.load_default()
        base_size = 12

    # measure characters and reduce font size until it fits
    chars = list(text)
    draw_tmp = ImageDraw.Draw(Image.new("RGBA", (10,10)))
    font_size = base_size
    inter_gap = 2
    while font_size >= 6:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
        total_h = 0
        max_w = 0
        heights = []
        for ch in chars:
            w, h = get_text_size(draw_tmp, ch, font)
            heights.append(h)
            total_h += h
            if w > max_w: max_w = w
        total_h += inter_gap * (len(chars)-1)
        if total_h <= max_height and max_w <= max_width:
            break
        font_size -= 1

    # create image and draw stacked characters centered horizontally
    pad = 2
    img_w = max(1, min(max_width, max_w + pad*2))
    img_h = max(1, min(max_height, total_h + pad*2))
    img = Image.new("RGBA", (img_w, img_h), (0,0,0,0))
    d = ImageDraw.Draw(img)
    y = pad
    for ch, h in zip(chars, heights):
        w, _ = get_text_size(d, ch, font)
        d.text(((img_w - w) / 2, y), ch, font=font, fill=(0,0,0,255))
        y += h + inter_gap

    return img

def compose_vertical_page(models_slice, page_index, output_dir):
    rows = len(models_slice)
    page_w = MODEL_NAME_W + COLS*CELL_W + (COLS-1)*H_SPACING
    page_h = TOP_MARGIN_H + rows*CELL_H + (rows-1)*V_SPACING
    canvas = Image.new("RGB", (page_w, page_h), (255,255,255))
    draw = ImageDraw.Draw(canvas)
    try:
        caption_font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        caption_font = ImageFont.load_default()

    col_labels = ["CONFUSION_MATRICES", "Bubbles", "EXTERNAL", "GRADCAM"]

    for row_idx, model_name in enumerate(models_slice):
        y_row = TOP_MARGIN_H + row_idx*(CELL_H + V_SPACING)

        # --- create narrow stacked vertical model-name image and paste centered vertically ---
        vert_img = create_stacked_vertical_text_image(model_name, MODEL_NAME_W, CELL_H)
        rw, rh = vert_img.size
        paste_x = int((MODEL_NAME_W - rw) / 2)
        paste_y = int(y_row + (CELL_H - rh) / 2)
        canvas.paste(vert_img, (paste_x, paste_y), vert_img)
        # --- end vertical model name ---

        imgs = find_csm_for_model(model_name)
        for col_idx in range(COLS):
            x_col = MODEL_NAME_W + col_idx*(CELL_W + H_SPACING)
            # shrink confusion matrices slightly
            if col_idx == 0:
                inner_w = int(CELL_W * CELL_SCALE_CM)
                inner_h = int(CELL_H * CELL_SCALE_CM)
            else:
                inner_w = CELL_W
                inner_h = CELL_H
            img_path = imgs[col_idx] if col_idx < len(imgs) else None
            if img_path:
                slot_img = load_and_fit(img_path, inner_w, inner_h)
            else:
                slot_img = make_placeholder(PLACEHOLDER_TEXT, inner_w, inner_h)
            paste_x = x_col + (CELL_W - slot_img.width)//2
            paste_y = y_row + (CELL_H - slot_img.height)//2
            canvas.paste(slot_img, (paste_x, paste_y))

            # draw border around the full cell
            left = x_col
            top = y_row
            right = x_col + CELL_W - 1
            bottom = y_row + CELL_H - 1
            for off in range(BORDER_WIDTH):
                draw.rectangle([left+off, top+off, right-off, bottom-off], outline=BORDER_COLOR)

            # small caption at bottom-right of cell (column label)
            label = col_labels[col_idx]
            lw, lh = get_text_size(draw, label, caption_font)
            draw.text((x_col + CELL_W - lw - 6, y_row + CELL_H - lh - 6), label, fill=(80,80,80), font=caption_font)

    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"merged_heatmaps_vertical_page_{page_index+1}.png")
    canvas.save(out_png, format="PNG")
    return out_png

def build_pdf(png_paths, pdf_outpath):
    pdf = FPDF(orientation='P', unit='mm', format='A4')  # portrait
    for png in png_paths:
        pdf.add_page()
        x = 10
        y = 10
        w_mm = pdf.w - 20
        pdf.image(png, x=x, y=y, w=w_mm)
    pdf.output(pdf_outpath)

if __name__ == "__main__":
    model_folders = [f for f in os.listdir(GRADCAM_OCC_OCC_DIR_PATH) if os.path.isdir(os.path.join(GRADCAM_OCC_OCC_DIR_PATH, f))]
    model_folders.sort(key=natural_sort_key)

    pages = [model_folders[i:i+ROWS] for i in range(0, len(model_folders), ROWS)]
    merged_pngs = []
    out_dir = MERGED_OCC_OCC_HEATMAPS_PDFS_DIR_PATH
    os.makedirs(out_dir, exist_ok=True)

    for pi, slice_models in enumerate(pages):
        print(f"Composing vertical page {pi+1} with {len(slice_models)} models...")
        merged = compose_vertical_page(slice_models, pi, out_dir)
        merged_pngs.append(merged)

    pdf_path = os.path.join(out_dir, "merged_heatmaps_vertical.pdf")
    print("Building PDF:", pdf_path)
    build_pdf(merged_pngs, pdf_path)
    print("Done.")