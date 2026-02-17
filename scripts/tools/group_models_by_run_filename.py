import os
import sys
from collections import defaultdict
from rich.console import Console
from rich.table import Table

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.config import OUT_ERR_DIR

# Constants
DIR_PATH = OUT_ERR_DIR
JOB_NAME = "dbuh_do_training_"

if __name__ == "__main__":
    # Initialize Rich console
    console = Console()

    # List all files in the directory
    filenames = os.listdir(DIR_PATH)
    
    # Filter files to include only .log or .txt files
    filenames = [f for f in filenames if f.endswith('.log') or f.endswith('.txt')]
    
    # Remove file extensions
    filenames = [f.strip(".log") if f.endswith('.log') else f.strip(".txt") for f in filenames]
    
    # Filter files that contain the JOB_NAME
    filenames = [f for f in filenames if JOB_NAME in f]

    # Remove the job name from the filenames
    filenames = [f.replace(JOB_NAME, "") for f in filenames]

    print(f"Found {len(filenames)} files matching the criteria in {DIR_PATH}.")

    # Extract information from filenames
    extracted_info = []
    for f in filenames:
        try:
            parts = f.split("_")
            file_id = parts[0]  # The ID is the first part of the filename
            model = parts[1]  # The model name comes after "do_training_"

            longep = "longep" in parts  # True if "longep" is in the parts
            occ = next((p for p in parts if p.startswith("occ")), None)
            occ = occ.split("occ")[1] if occ else None  # Extract the value after "occ"
            unf = next((p for p in parts if p.startswith("unf")), None)
            unf = unf.split("unf")[1] if unf else None  # Extract the value after "unf"
            acc = next((p for p in parts if p.startswith("acc")), None)
            acc = acc.split("acc")[1] if acc else None  # Extract the value after "acc"

            extracted_info.append({
                "id": file_id,
                "model": model,
                "occ": occ,
                "unf": unf,
                "longep": longep,
                "acc": acc
            })
        except Exception as e:
            print(f"Error processing file {f}: {e}")


    # Group data by model and by (occ, unf, longep)
    grouped_data = defaultdict(lambda: defaultdict(list))
    for info in extracted_info:
        key = (info["occ"], info["unf"], info["longep"])
        grouped_data[info["model"]][key].append(info)

    # Print the table for each model
    print(f"=============== Grouped result by model ================================================")
    for model, entries in grouped_data.items():
        # Create a Rich table
        table = Table(title=f"Model: {model}")
        table.add_column("OCC", justify="center")
        table.add_column("UNF", justify="center")
        table.add_column("LONGEP", justify="center")
        table.add_column("IDs", justify="center")
        table.add_column("ACCs", justify="center")

        # Sort entries by the best accuracy value in descending order
        sorted_entries = sorted(
            entries.items(),
            key=lambda item: min(float(info["acc"]) if info["acc"] else 0 for info in item[1]),
            reverse=True
        )

        # Add rows to the table
        for (occ, unf, longep), infos in sorted_entries:
            # Sort infos by accuracy (ascending order)
            infos = sorted(infos, key=lambda info: float(info["acc"]) if info["acc"] else 0)

            ids = ", ".join(info["id"] for info in infos)
            accs = ", ".join(f"0.{info['acc']}" if info["acc"] else "N/A" for info in infos)
            table.add_row(
                f"0.{occ}" if occ else "N/A",
                unf if unf else "N/A",
                str(longep),
                ids,
                accs
            )

        # Print the table
        console.print(table)
    print(f"========================================================================================")

    # Create a table for the best result of each model
    best_results = []
    for model, entries in grouped_data.items():
        # Get the best entry (highest accuracy) for each model
        best_entry = max(
            (info for infos in entries.values() for info in infos),
            key=lambda x: x["acc"]
        )
        best_results.append({
            "model": model,
            "occ": best_entry["occ"],
            "unf": best_entry["unf"],
            "longep": best_entry["longep"],
            "acc": best_entry["acc"]
        })

    # Sort the best results by accuracy (descending order)
    best_results.sort(key=lambda x: x["acc"], reverse=True)

    # Create a Rich table for the best results
    best_table = Table(title="Best Results by Model (Ranked by Accuracy)")
    best_table.add_column("MODEL", justify="center")
    best_table.add_column("OCC", justify="center")
    best_table.add_column("UNF", justify="center")
    best_table.add_column("LONGEP", justify="center")
    best_table.add_column("ACC", justify="center")

    # Add rows to the best results table
    for result in best_results:
        best_table.add_row(
            result["model"],
            f"0.{result['occ']}" if result["occ"] else "N/A",
            result["unf"] if result["unf"] else "N/A",
            str(result["longep"]),
            f"0.{result['acc']}" if result["acc"] else "N/A"
        )

    # Print the best results table
    console.print(best_table)