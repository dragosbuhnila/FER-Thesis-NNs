import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from PIL import Image
from fpdf import FPDF
import re
import unicodedata

from modules.config import GRADCAM_DIR_PATH, GRADCAM_LAYERS_PDFS_DIR_PATH


def layer_sort_key(layer_folder_name):
    match = re.search(r'layer_(\d+)', layer_folder_name)
    if match:
        return int(match.group(1))
    else:
        return float('inf')


def make_caption(model_name, layer_folder):
    display_name = 'mobilenet' if 'pattlite' in model_name.lower() else model_name
    layer_folder
    return f"GRADCAM - {display_name} - {layer_folder}"

def safe_str_for_pdf(s):
    # Replace common problematic Unicode and reduce to ASCII
    s = s.replace('\u2014', ' - ').replace('\u2013', ' - ')
    # Normalize and drop anything non-ASCII
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    return s


def create_pdf_for_model(model_folder, output_dir, debug=False):
    model_path = os.path.join(GRADCAM_DIR_PATH, model_folder)
    layer_folders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
    layer_folders.sort(key=layer_sort_key)

    pdf = FPDF(unit='mm', format='A4')
    pdf.set_auto_page_break(False)  # prevent automatic page breaks that push captions to the next page
    left_margin = 12
    right_margin = 12
    top_margin = 12
    bottom_margin = 12
    between_images_spacing = 8  # vertical spacing between image slots
    caption_height = 8  # space reserved for caption under each image (mm)
    caption_font_size = 10
    images_per_page = 2
    image_count = 0

    page_width = pdf.w
    page_height = pdf.h
    available_width = page_width - left_margin - right_margin
    available_height = page_height - top_margin - bottom_margin
    slot_height = (available_height - between_images_spacing) / 2.0
    max_image_height = slot_height - caption_height - 2.0  # extra small gap

    for layer_folder in layer_folders:
        if debug and image_count > 3:
            break
        print(f"\tProcessing layer folder: {layer_folder}")

        csm_path = os.path.join(model_path, layer_folder, "CSM.png")
        if not os.path.exists(csm_path):
            print(f"CSM.png not found in {layer_folder}. Skipping.")
            continue

        try:
            img = Image.open(csm_path)
        except Exception as e:
            print(f"Failed to open {csm_path}: {e}. Skipping.")
            continue

        img_w_px, img_h_px = img.size
        if img_w_px == 0 or img_h_px == 0:
            print(f"Invalid image size for {csm_path}. Skipping.")
            continue
        aspect = img_h_px / img_w_px

        # determine target size within available width and max_image_height
        target_w = available_width
        target_h = target_w * aspect
        if target_h > max_image_height:
            target_h = max_image_height
            target_w = target_h / aspect

        # Start new page when required
        if image_count % images_per_page == 0:
            pdf.add_page()
            # set default font for caption
            pdf.set_font("Arial", style='I', size=caption_font_size)

        slot_index = image_count % images_per_page  # 0 = top slot, 1 = bottom slot
        x_image = left_margin + (available_width - target_w) / 2.0
        slot_top = top_margin + slot_index * (slot_height + between_images_spacing)
        y_image = slot_top + ( (max_image_height - target_h) / 2.0 )

        pdf.image(csm_path, x=x_image, y=y_image, w=target_w, h=target_h)

        # caption below image
        caption_text = make_caption(model_folder, layer_folder)
        y_caption = y_image + target_h + 2.0  # small gap between image and caption
        # ensure caption stays within slot
        if y_caption + caption_height > slot_top + slot_height:
            y_caption = slot_top + slot_height - caption_height

        caption_text = make_caption(model_folder, layer_folder)
        caption_text = safe_str_for_pdf(caption_text)

        pdf.set_xy(left_margin, y_caption)
        pdf.set_font("Arial", style='I', size=caption_font_size)
        pdf.cell(w=available_width, h=caption_height, txt=caption_text, border=0, ln=0, align='C')

        image_count += 1

    # Save PDF
    os.makedirs(output_dir, exist_ok=True)
    # replace pattlite to mobilenet in filename
    model_folder = model_folder.replace("pattlite", "mobilenet") if "pattlite" in model_folder.lower() else model_folder
    output_pdf_path = os.path.join(output_dir, f"{model_folder}_CSM_layers.pdf")
    pdf.output(output_pdf_path)
    print(f"PDF created: {output_pdf_path}")


if __name__ == "__main__":
    model_folders = [f for f in os.listdir(GRADCAM_DIR_PATH) if os.path.isdir(os.path.join(GRADCAM_DIR_PATH, f))]
    output_dir = GRADCAM_LAYERS_PDFS_DIR_PATH

    for model_folder in model_folders:
        print(f"Processing model: {model_folder}")
        create_pdf_for_model(model_folder, output_dir, debug=False)
