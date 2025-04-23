import configparser
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re # Keep import
import seaborn as sns # Import seaborn for better aesthetics

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
    print(f"Error: Missing required keys in [paths] section of '{config_file_path}': {missing}")
    exit()

RESULT_PATH = config["paths"]["result_path"]
VISUALIZATION_PATH = config["paths"]["visualization_path"]
os.makedirs(VISUALIZATION_PATH, exist_ok=True) # Ensure visualization dir exists

# --- Define Datasets and Models ---
datasets = ['college', 'music', 'movie']
model_sizes = ['1b', '4b', '12b']
model_names_map = { # Map size to display name
    '1b': 'Gemma 1B',
    '4b': 'Gemma 4B',
    '12b': 'Gemma 12B'
}
ordered_model_names = [model_names_map[size] for size in model_sizes] # For consistent column order

# --- Desired Attribute Order (Used for Extraction) ---
attribute_order_nested = {
    "country": ["an American", "a Brazilian", "a British", "a Chinese", "a French", "a German", "a Japanese"],
    "gender": ["a girl", "a boy", "a female", "a male"],
    "continent": ["an African", "an American", "an Asian"],
    "occupation": ["a child of a doctor", "a child of a student", "a child of a teacher", "a child of a worker", "a child of a writer"],
    "race": ["a Black", "a White", "an Asian", "a Hispanic"],
    "religion": ["a Buddhist", "a Christian", "a Muslim"],
    "physics": ["fat", "thin"]
}
attribute_order = []
category_order = ["country", "gender", "continent", "occupation", "race", "religion", "physics"]
for category in category_order:
    if category in attribute_order_nested:
        attribute_order.extend(attribute_order_nested[category])
    else:
        print(f"Warning: Category '{category}' from category_order not found in attribute_order_nested.")

# --- Updated Helper Function ---
def extract_iou_means_flat(filename, attributes_to_extract):
    """
    Loads JSON data and extracts the mean 'IOU Divergence' for specified attributes.
    """
    iou_data = {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        extracted_count = 0
        for attribute in attributes_to_extract:
            attr_data = data.get(attribute)
            if not attr_data or not isinstance(attr_data, dict):
                iou_data[attribute] = np.nan
                continue

            # *** UPDATED KEY HERE ***
            iou_dict = attr_data.get('IOU Divergence') # Changed from 'IOU'

            if not iou_dict or not isinstance(iou_dict, dict):
                iou_data[attribute] = np.nan
                continue

            mean_iou = iou_dict.get('mean')
            if mean_iou is None:
                iou_data[attribute] = np.nan
            else:
                iou_data[attribute] = mean_iou
                extracted_count += 1

        if extracted_count == 0:
             print(f"  Warning: No 'IOU Divergence' -> 'mean' values found in {os.path.basename(filename)} for the requested attributes.")


    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return None # Indicate file not found error
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filename}")
        return {} # Return empty dict on JSON error, allows processing to continue
    except Exception as e:
        print(f"An unexpected error occurred processing {filename}: {e}")
        return {} # Return empty dict on other errors

    # Return data ensuring all requested keys are present
    final_data = {key: iou_data.get(key, np.nan) for key in attributes_to_extract}
    return final_data

# --- Load Data and Calculate Overall Mean IOU Divergence ---
overall_mean_ious = {}
all_files_found = True

print("--- Processing Files ---")
for dataset in datasets:
    overall_mean_ious[dataset.capitalize()] = {} # Use capitalized dataset name for display
    print(f"Dataset: {dataset}")
    for model_size in model_sizes:
        model_display_name = model_names_map[model_size]
        filename = os.path.join(RESULT_PATH, f'gemma3_{model_size}_{dataset}_.json')
        print(f"  Model: {model_display_name} - File: {os.path.basename(filename)}")

        if not os.path.exists(filename):
            print(f"    -> File NOT FOUND.")
            overall_mean_ious[dataset.capitalize()][model_display_name] = np.nan
            all_files_found = False
            continue # Skip to next model if file is missing

        # Extract IOU means for all attributes for this specific file
        iou_means_dict = extract_iou_means_flat(filename, attribute_order)

        if iou_means_dict is None: # Specific check for file not found from function
             overall_mean_ious[dataset.capitalize()][model_display_name] = np.nan
             all_files_found = False # Mark that at least one file was missing
             continue

        # Calculate the mean of the extracted means, ignoring NaNs
        valid_iou_values = [v for v in iou_means_dict.values() if pd.notna(v)]
        if valid_iou_values:
            mean_of_means = np.mean(valid_iou_values)
            overall_mean_ious[dataset.capitalize()][model_display_name] = mean_of_means
            print(f"    -> Overall Mean IOU Divergence: {mean_of_means:.4f} (from {len(valid_iou_values)} attributes)")
        else:
            print(f"    -> No valid IOU Divergence data found for any attribute.")
            overall_mean_ious[dataset.capitalize()][model_display_name] = np.nan

if not all_files_found:
    print("\nWarning: One or more result files were not found. The plot will show missing data (NaNs).")

# --- Prepare DataFrame for Plotting ---
# Create DataFrame with Datasets as index and Models as columns
df_plot = pd.DataFrame(overall_mean_ious).T # Transpose to get datasets as rows

# Ensure columns are in the desired order (1B, 4B, 12B)
df_plot = df_plot.reindex(columns=ordered_model_names)

print("\n--- Data for Plotting ---")
print(df_plot.to_string(float_format="%.4f"))

# --- Create Visualization (Grouped Bar Chart) ---
if df_plot.isnull().all().all():
    print("\nError: No valid data available to plot after processing all files. Exiting.")
    exit()
elif df_plot.isnull().any().any():
     print("\nNote: Plot contains missing data (NaNs) due to file issues or lack of valid IOU Divergence values.")


plt.style.use('seaborn-v0_8-colorblind') # Use a visually appealing style
fig, ax = plt.subplots(figsize=(12, 7))

df_plot.plot(kind='bar', ax=ax, width=0.8) # Use pandas plotting

ax.set_xlabel("Dataset", fontsize=12)
ax.set_ylabel("Overall Mean IOU Divergence", fontsize=12)
ax.set_title("Overall Mean IOU Divergence per Model Across Datasets", fontsize=14, pad=15)
ax.tick_params(axis='x', rotation=0) # Keep dataset names horizontal
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(title="Model Size", fontsize=10, title_fontsize=11)

# Adjust y-axis limits slightly for padding if data exists
if not df_plot.dropna().empty:
    min_val = df_plot.min().min()
    max_val = df_plot.max().max()
    # Ensure min_val is not NaN before calculation
    if pd.notna(min_val) and pd.notna(max_val):
         ax.set_ylim(bottom=max(0, min_val - (max_val-min_val)*0.1),
                     top=max_val + (max_val-min_val)*0.1)
    else:
         ax.set_ylim(bottom=0) # Default if only NaNs
else:
    ax.set_ylim(bottom=0)


plt.tight_layout(pad=1.5)

# --- Save and Show Plot ---
output_plot_file = os.path.join(VISUALIZATION_PATH, "overall_iou_divergence_by_model_dataset.png")

try:
    plt.savefig(output_plot_file, dpi=300)
    print(f"\nPlot saved successfully to: {output_plot_file}")
except Exception as e:
    print(f"\nError saving plot: {e}")

plt.show()

print("\n--- Interpretation ---")
print("This bar chart shows the average 'IOU Divergence' across all sensitive attributes for each Gemma model size, grouped by dataset.")
print("Each group on the x-axis represents a dataset (College, Music, Movie).")
print("Within each group, the bars represent Gemma 1B, 4B, and 12B models.")
print("The height of each bar indicates the overall mean IOU Divergence for that specific model on that specific dataset.")
print("Higher bars suggest greater average divergence in recommendations based on sensitive attributes for that model/dataset combination.")
print("Missing bars indicate that the corresponding result file was not found or contained no valid 'IOU Divergence' data.")