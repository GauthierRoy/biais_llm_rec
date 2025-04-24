import configparser
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re  # Keep import
import seaborn as sns  # Import seaborn for better aesthetics

# --- Configuration ---
config = configparser.ConfigParser()
config_file_path = "config/config_inference"
if not os.path.exists(config_file_path):
    print(f"Error: Configuration file '{config_file_path}' not found.")
    exit()
config.read(config_file_path)

required_paths = {"output_path", "result_path", "visualization_path"}
if "paths" not in config or not required_paths.issubset(config["paths"]):
    missing = required_paths - set(config.options("paths") if "paths" in config else [])
    print(
        f"Error: Missing required keys in [paths] section of '{config_file_path}': {missing}"
    )
    exit()

RESULT_PATH = config["paths"]["result_path"]
VISUALIZATION_PATH = config["paths"]["visualization_path"]
os.makedirs(VISUALIZATION_PATH, exist_ok=True)  # Ensure visualization dir exists

# --- Define Datasets and Models ---
datasets = ["college", "music", "movie"]
# Specific models to compare
model_ids = ["gemma3_4b", "llama3.2_3b"]
model_names_map = {  # Map model ID to display name
    "gemma3_4b": "Gemma 3 4B",
    "llama3.2_3b": "Llama 3.2 3B",
}
ordered_model_names = [
    model_names_map[model_id] for model_id in model_ids
]  # For consistent column order

# --- Desired Attribute Order (Used for Extraction) ---
attribute_order_nested = {
    "country": [
        "an American",
        "a Brazilian",
        "a British",
        "a Chinese",
        "a French",
        "a German",
        "a Japanese",
    ],
    "gender": ["a girl", "a boy", "a female", "a male"],
    "continent": ["an African", "an American", "an Asian"],
    "occupation": [
        "a child of a doctor",
        "a child of a student",
        "a child of a teacher",
        "a child of a worker",
        "a child of a writer",
    ],
    "race": ["a Black", "a White", "an Asian", "a Hispanic"],
    "religion": ["a Buddhist", "a Christian", "a Muslim"],
    "physics": ["fat", "thin"],
}
attribute_order = []
category_order = [
    "country",
    "gender",
    "continent",
    "occupation",
    "race",
    "religion",
    "physics",
]
for category in category_order:
    if category in attribute_order_nested:
        attribute_order.extend(attribute_order_nested[category])
    else:
        print(
            f"Warning: Category '{category}' from category_order not found in attribute_order_nested."
        )


# --- Updated Helper Function ---
def extract_metric_means_flat(filename, attributes_to_extract, metric="IOU Divergence"):
    f"""
    Loads JSON data and extracts the mean {metric} for specified attributes.
    """
    iou_data = {}
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        extracted_count = 0
        for attribute in attributes_to_extract:
            attr_data = data.get(attribute)
            if not attr_data or not isinstance(attr_data, dict):
                iou_data[attribute] = np.nan
                continue

            iou_dict = attr_data.get(metric)

            if not iou_dict or not isinstance(iou_dict, dict):
                iou_data[attribute] = np.nan
                continue

            mean_iou = iou_dict.get("mean")
            if mean_iou is None:
                iou_data[attribute] = np.nan
            else:
                iou_data[attribute] = mean_iou
                extracted_count += 1

        if extracted_count == 0:
            print(
                f"  Warning: No {metric} -> 'mean' values found in {os.path.basename(filename)} for the requested attributes."
            )

    except FileNotFoundError:
        # Return None specifically for file not found to handle it in the main loop
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filename}")
        return {}  # Return empty dict on JSON error, allows processing to continue
    except Exception as e:
        print(f"An unexpected error occurred processing {filename}: {e}")
        return {}  # Return empty dict on other errors

    # Return data ensuring all requested keys are present
    final_data = {key: iou_data.get(key, np.nan) for key in attributes_to_extract}
    return final_data


# --- Load Data and Calculate Overall Mean metric ---
overall_mean_ious = {}
all_files_found = True

print("--- Processing Files ---")
metrics = ["IOU Divergence", "SERP MS Divergence", "Pragmatic Divergence"]
for metric in metrics:

    for dataset in datasets:
        overall_mean_ious[dataset.capitalize()] = (
            {}
        )  # Use capitalized dataset name for display
        print(f"Dataset: {dataset}")
        for model_id in model_ids:
            model_display_name = model_names_map[model_id]
            # Construct filename using the model ID directly
            filename = os.path.join(RESULT_PATH, f"{model_id}_{dataset}_.json")
            print(f"  Model: {model_display_name} - File: {os.path.basename(filename)}")

            # Check file existence before calling extraction function
            if not os.path.exists(filename):
                print(f"    -> File NOT FOUND.")
                overall_mean_ious[dataset.capitalize()][model_display_name] = np.nan
                all_files_found = False
                continue  # Skip to next model if file is missing

            # Extract IOU means for all attributes for this specific file
            iou_means_dict = extract_metric_means_flat(
                filename, attribute_order, metric=metric
            )

            if iou_means_dict is None:  # Check for file not found return value
                print(
                    f"    -> Error processing file (likely not found, handled above)."
                )
                overall_mean_ious[dataset.capitalize()][model_display_name] = np.nan
                # all_files_found is already False from the os.path.exists check
                continue
            elif (
                not iou_means_dict
            ):  # Check for empty dict return value (JSON error, etc.)
                print(f"    -> Error processing file (JSON decode or other issue).")
                overall_mean_ious[dataset.capitalize()][model_display_name] = np.nan
                continue

            # Calculate the mean of the extracted means, ignoring NaNs
            valid_iou_values = [v for v in iou_means_dict.values() if pd.notna(v)]
            if valid_iou_values:
                mean_of_means = np.mean(valid_iou_values)
                overall_mean_ious[dataset.capitalize()][
                    model_display_name
                ] = mean_of_means
                print(
                    f"    -> Overall Mean {metric}: {mean_of_means:.4f} (from {len(valid_iou_values)} attributes)"
                )
            else:
                # This case handles when the file exists and is valid JSON, but no metric ->'mean' keys were found
                print(f"    -> No valid {metric} data found for any attribute.")
                overall_mean_ious[dataset.capitalize()][model_display_name] = np.nan

    if not all_files_found:
        print(
            "\nWarning: One or more result files were not found. The plot will show missing data (NaNs)."
        )

    # --- Prepare DataFrame for Plotting ---
    # Create DataFrame with Datasets as index and Models as columns
    df_plot = pd.DataFrame(overall_mean_ious).T  # Transpose to get datasets as rows

    # Ensure columns are in the desired order (Gemma, Llama)
    df_plot = df_plot.reindex(columns=ordered_model_names)

    print("\n--- Data for Plotting ---")
    print(df_plot.to_string(float_format="%.4f"))

    # --- Create Visualization (Grouped Bar Chart) ---
    if df_plot.isnull().all().all():
        print(
            "\nError: No valid data available to plot after processing all files. Exiting."
        )
        exit()
    elif df_plot.isnull().any().any():
        print(
            f"\nNote: Plot contains missing data (NaNs) due to file issues or lack of valid {metric} values."
        )

    plt.style.use("seaborn-v0_8-colorblind")  # Use a visually appealing style
    fig, ax = plt.subplots(figsize=(12, 7))

    df_plot.plot(kind="bar", ax=ax, width=0.6)  # Adjust width if needed for two bars

    ax.set_xlabel("Dataset", fontsize=14)  # Increased fontsize
    ax.set_ylabel(f"Overall Mean {metric}", fontsize=14)  # Increased fontsize
    ax.set_title(
        f"Overall Mean {metric}: Gemma 3 4B vs Llama 3.2 3B Across Datasets",
        fontsize=16,
        pad=15,
    )  # Increased fontsize
    ax.tick_params(
        axis="x", rotation=0, labelsize=12
    )  # Keep dataset names horizontal, increased labelsize
    ax.tick_params(axis="y", labelsize=12)  # Increased y-axis tick labelsize
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(
        title="Model", fontsize=12, title_fontsize=13
    )  # Increased legend font sizes

    # Adjust y-axis limits slightly for padding if data exists
    if not df_plot.dropna().empty:
        # Calculate min/max across the entire DataFrame, ignoring NaNs
        min_val = df_plot.min(skipna=True).min(skipna=True)
        max_val = df_plot.max(skipna=True).max(skipna=True)

        # Ensure min_val and max_val are not NaN before calculation
        if pd.notna(min_val) and pd.notna(max_val):
            # Add padding, ensuring bottom limit is not below 0
            padding = (
                (max_val - min_val) * 0.1 if max_val > min_val else 0.1
            )  # Handle case where min=max
            ax.set_ylim(bottom=max(0, min_val - padding), top=max_val + padding)
        elif pd.notna(
            max_val
        ):  # Handle case where only max_val is valid (e.g., all NaNs except one value)
            ax.set_ylim(bottom=0, top=max_val + 0.1)
        else:  # Handle case where all values might be NaN
            ax.set_ylim(bottom=0)  # Default if only NaNs
    else:
        ax.set_ylim(bottom=0)

    plt.tight_layout(pad=1.5)

    # --- Save and Show Plot ---
    output_plot_file = os.path.join(
        VISUALIZATION_PATH, "overall_iou_divergence_gemma_vs_llama.png"
    )

    try:
        plt.savefig(output_plot_file, dpi=300)
        print(f"\nPlot saved successfully to: {output_plot_file}")
    except Exception as e:
        print(f"\nError saving plot: {e}")

    plt.show()

    print("\n--- Interpretation ---")
    print(
        f"This bar chart compares the average {metric} across all sensitive attributes for Gemma 3 4B and Llama 3.2 3B models, grouped by dataset."
    )
    print("Each group on the x-axis represents a dataset (College, Music, Movie).")
    print(
        "Within each group, the bars represent the Gemma 3 4B and Llama 3.2 3B models."
    )
    print(
        f"The height of each bar indicates the overall mean {metric} for that specific model on that specific dataset."
    )
    print(
        "Higher bars suggest greater average divergence in recommendations based on sensitive attributes for that model/dataset combination."
    )
    print(
        f"Missing bars indicate that the corresponding result file was not found or contained no valid {metric} data."
    )
