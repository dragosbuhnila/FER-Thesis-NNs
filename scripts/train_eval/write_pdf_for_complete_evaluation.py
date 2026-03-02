import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from modules.config import SAVED_IMAGES_PATH, PROJECT_ROOT


def create_pdf(output_path, base_folder, agreement_folder_name="agreement_analysis", model_folder_signatures=["occft", 'finetuning']):
    print(f"[INFO] Starting PDF generation. Output path: {output_path}")
    print(f"[INFO] Base folder: {base_folder}")

    # Initialize the PDF
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    body_style = styles['BodyText']

    # Add a title
    elements.append(Paragraph("Model Analysis Report", title_style))
    elements.append(Spacer(1, 20))

    # Process the agreement_analysis folder
    agreement_folder = os.path.join(base_folder, agreement_folder_name)
    if os.path.exists(agreement_folder):
        print(f"[INFO] Processing agreement_analysis folder: {agreement_folder}")
        elements.append(Paragraph("Agreement Analysis", heading_style))
        elements.append(Spacer(1, 10))

        # Add CSVs as tables
        for csv_file in ["agreement_statistics.csv", "agreement_values.csv"]:
            csv_path = os.path.join(agreement_folder, csv_file)
            if os.path.exists(csv_path):
                print(f"[INFO] Adding CSV file to PDF: {csv_path}")
                elements.append(Paragraph(f"Table: {csv_file}", body_style))
                elements.append(Spacer(1, 5))
                add_csv_to_pdf(csv_path, elements)
            else:
                print(f"[WARNING] CSV file not found: {csv_path}")

        # Add images
        disagreeing_images_folder = os.path.join(agreement_folder, "disagreeing_images")
        if os.path.exists(disagreeing_images_folder):
            print(f"[INFO] Processing disagreeing_images folder: {disagreeing_images_folder}")
            i = 0
            for img_file in sorted(os.listdir(disagreeing_images_folder)):
                img_path = os.path.join(disagreeing_images_folder, img_file)
                if img_file.endswith(".png"):
                    print(f"[INFO] Adding image to PDF: {img_path}")
                    elements.append(Paragraph(f"Image: {img_file}", body_style))
                    elements.append(Spacer(1, 5))
                    add_image_to_pdf(img_path, elements)
                    i += 1
                    if i >= 4:  # Limit to 4 images to avoid overcrowding the PDF
                        print(f"[WARNING] Reached image limit for disagreeing_images. Stopping further additions.")
                        break
                else:
                    print(f"[WARNING] Skipping non-PNG file: {img_file}")
        else:
            print(f"[WARNING] disagreeing_images folder not found: {disagreeing_images_folder}")

    # Process each model folder
    model_folders = [f for f in os.listdir(base_folder) if any(sig in f for sig in model_folder_signatures) and os.path.isdir(os.path.join(base_folder, f))]
    print(f"[INFO] Found {len(model_folders)} model folders: {model_folders}")
    for model_folder in model_folders:
        model_path = os.path.join(base_folder, model_folder)
        print(f"[INFO] Processing model folder: {model_folder}")
        elements.append(Paragraph(f"Model: {model_folder}", heading_style))
        elements.append(Spacer(1, 10))

        # Add CSVs as tables
        for csv_file in ["accuracies.csv", "precision_recall_f1.csv", "probs_ytrue_ypred.csv"]:
            csv_path = os.path.join(model_path, csv_file)
            if os.path.exists(csv_path):
                print(f"[INFO] Adding CSV file to PDF: {csv_path}")
                elements.append(Paragraph(f"Table: {csv_file}", body_style))
                elements.append(Spacer(1, 5))
                add_csv_to_pdf(csv_path, elements)
            else:
                print(f"[WARNING] CSV file not found: {csv_path}")

        # Add images from subfolders
        for subfolder in ["confusion_matrix", "high_confidence_errors", "tsne", "uncertain_predictions"]:
            subfolder_path = os.path.join(model_path, subfolder)
            if os.path.exists(subfolder_path):
                i = 0
                print(f"[INFO] Processing subfolder: {subfolder_path}")
                elements.append(Paragraph(f"Images from {subfolder}:", body_style))
                elements.append(Spacer(1, 5))
                for img_file in sorted(os.listdir(subfolder_path)):
                    img_path = os.path.join(subfolder_path, img_file)
                    if img_file.endswith(".png"):
                        print(f"[INFO] Adding image to PDF: {img_path}")
                        elements.append(Paragraph(f"Image: {img_file}", body_style))
                        elements.append(Spacer(1, 5))
                        add_image_to_pdf(img_path, elements)
                        i += 1
                        if i >= 4:  # Limit to 4 images to avoid overcrowding the PDF
                            print(f"[WARNING] Reached image limit for subfolder {subfolder}. Stopping further additions.")
                            break
                    else:
                        print(f"[WARNING] Skipping non-PNG file: {img_file}")
            else:
                print(f"[WARNING] Subfolder not found: {subfolder_path}")

    # Build the PDF
    print(f"[INFO] Building the PDF with {len(elements)} elements.")
    doc.build(elements)
    print(f"[INFO] PDF generation completed. File saved at: {output_path}")


def add_csv_to_pdf(csv_path, elements, max_rows=40, max_columns=8):
    """Add a CSV file as a table to the PDF, shrinking wide tables and limiting numerical precision."""
    try:
        print(f"[WARNING] Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)

        # Limit rows to max_rows
        if len(df) > max_rows:
            print(f"[WARNING] Truncating CSV to {max_rows} rows.")
            df = df.head(10)

        # Format numerical columns to 4 digits (not decimal places, but total digits)
        for col in df.select_dtypes(include=['float', 'int']).columns:
            df[col] = df[col].apply(lambda x: f"{x:.4g}" if pd.notnull(x) else x)

        # Exclude Image_Path column if it exists to avoid cluttering the table with long paths
        if 'Image_Path' in df.columns:
            print(f"[WARNING] Excluding 'Image_Path' column from table to avoid clutter.")
            df = df.drop(columns=['Image_Path'])

        # Prepare data for the table
        data = [df.columns.tolist()] + df.values.tolist()

        # Adjust column widths if there are too many columns
        num_columns = len(df.columns)
        if num_columns > max_columns:
            print(f"[WARNING] Adjusting column widths for {num_columns} columns.")
            col_widths = [600 / num_columns] * num_columns  # Shrink columns proportionally
        else:
            col_widths = None  # Default column widths

        # Create the table
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))
    except Exception as e:
        print(f"[ERROR] Failed to process CSV file {csv_path}: {e}")


def add_image_to_pdf(img_path, elements, max_width=600):
    """Add an image to the PDF."""
    try:
        print(f"[INFO] Adding image: {img_path}")
        img = Image(img_path)
        img._restrictSize(max_width, max_width)  # Restrict image size
        elements.append(img)
        elements.append(Spacer(1, 20))
    except Exception as e:
        print(f"[ERROR] Failed to add image {img_path}: {e}")


def extract_evaluation_type(model_folder_name):
    """Extract evaluation type from model folder name."""
    evaluation_name = ""

    if "cmlpt-run" in model_folder_name:
        evaluation_name += "cmplt-run_"
    elif "quick-run" in model_folder_name:
        evaluation_name += "quick-run_"
    
    if "occft" in model_folder_name:
        evaluation_name += "occft-models_"
    elif "federica" in model_folder_name:
        evaluation_name += "federica-models_"

    if "occluded-testset" in model_folder_name:
        evaluation_name += "occluded-testset"
    elif "original-testset" in model_folder_name:
        evaluation_name += "original-testset"
    elif "original-180-testset" in model_folder_name:
        evaluation_name += "original-180-testset"

    return evaluation_name


argparser = argparse.ArgumentParser(description="Generate a PDF report for model analysis.")
argparser.add_argument("--folder_name",         type=str, required=True,    help="Name of the folder to analyze (e.g., '20260220-164405_cmplt-run_occft-models_occluded-testset_do-evaluation-completely-keras').")
argparser.add_argument("--base_folder_path",    type=str, required=False,   help="Path to the base folder containing agreement_analysis and model subfolders.")
argparser.add_argument("--output_folder_path",  type=str, required=False,   help="Path to save the generated PDF report (default: current directory).")
args = argparser.parse_args()

if args.base_folder_path:
    BASE_FOLDER = os.path.join(args.base_folder_path, args.folder_name)
else:
    BASE_FOLDER = os.path.join(SAVED_IMAGES_PATH, args.folder_name)

OUTPUT_FILE_NAME = f"model_analysis_report_{extract_evaluation_type(os.path.basename(BASE_FOLDER))}.pdf"
if args.output_folder_path:
    OUTPUT_PATH = os.path.join(args.output_folder_path, OUTPUT_FILE_NAME)
else:
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, OUTPUT_FILE_NAME)

# Example usage:
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/write_pdf_for_complete_evaluation.py" --folder_name 20260303-003326_cmplt-run_occft-models_occluded-testset_do-evaluation-completely-keras
if __name__ == "__main__":
    create_pdf(OUTPUT_PATH, BASE_FOLDER, agreement_folder_name="agreement_analysis", model_folder_signatures=["occft", 'finetuning'])
    print(f"[INFO] PDF report generated: {OUTPUT_PATH}")