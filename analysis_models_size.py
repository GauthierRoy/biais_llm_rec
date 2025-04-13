import configparser
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re # Keep import

# --- Configuration ---
config = configparser.ConfigParser()
config_file_path = "config_inference"
if not os.path.exists(config_file_path):
    print(f"Error: Configuration file '{config_file_path}' not found.")
    exit()
config.read(config_file_path)

# Check for necessary paths in config
required_paths = {"output_path", "result_path", "visualization_path"}
if "paths" not in config or not required_paths.issubset(config["paths"]):
    missing = required_paths - set(config.options("paths") if "paths" in config else [])
    print(f"Error: Missing required keys in [paths] section of '{config_file_path}': {missing}")
    exit()

OUTPUT_PATH = config["paths"]["output_path"]
RESULT_PATH = config["paths"]["result_path"]
VISUALIZATION_PATH = config["paths"]["visualization_path"]

import configparser
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re # Keep import

# --- Configuration ---
config = configparser.ConfigParser()
config_file_path = "config_inference"
if not os.path.exists(config_file_path):
    print(f"Error: Configuration file '{config_file_path}' not found.")
    exit()
config.read(config_file_path)

# Check for necessary paths in config
required_paths = {"output_path", "result_path", "visualization_path"}
if "paths" not in config or not required_paths.issubset(config["paths"]):
    missing = required_paths - set(config.options("paths") if "paths" in config else [])
    print(f"Error: Missing required keys in [paths] section of '{config_file_path}': {missing}")
    exit()

OUTPUT_PATH = config["paths"]["output_path"]
RESULT_PATH = config["paths"]["result_path"]
VISUALIZATION_PATH = config["paths"]["visualization_path"]

# Define model files using RESULT_PATH
file_mapping = {
    "Gemma 12B": os.path.join(RESULT_PATH, 'gemma3_12b_college_.json'),
    # Corrected filename for 4B based on your provided sample filename
    "Gemma 4B": os.path.join(RESULT_PATH, 'gemma3_college_.json'),
    "Gemma 1B": os.path.join(RESULT_PATH, 'gemma3_1b_college_.json')
}
model_names = list(file_mapping.keys())
visualization_dir = VISUALIZATION_PATH # Use path from config

# --- Desired Attribute Order ---
# Nested dictionary is only used to define the final flat order
attribute_order_nested = {
    "country": ["an American", "a Brazilian", "a British", "a Chinese", "a French", "a German", "a Japanese"],
    "gender": ["a girl", "a boy", "a female", "a male"],
    "continent": ["an African", "an American", "an Asian"],
    "occupation": ["a child of a doctor", "a child of a student", "a child of a teacher", "a child of a worker", "a child of a writer"],
    "race": ["a Black", "a White", "an Asian", "a Hispanic"],
    "religion": ["a Buddhist", "a Christian", "a Muslim"],
    "physics": ["fat", "thin"]
}

# Flatten the list to define the desired plotting order
attribute_order = []
category_order = ["country", "gender", "continent", "occupation", "race", "religion", "physics"]
for category in category_order:
    if category in attribute_order_nested:
        attribute_order.extend(attribute_order_nested[category])
    else:
        print(f"Warning: Category '{category}' from category_order not found in attribute_order_nested.")

# --- ** CORRECTED ** Helper Function for Flat Data Structure ---
def extract_iou_means_flat(filename, attributes_to_extract):
    """
    Loads JSON data where attributes are top-level keys
    and extracts the mean IOU for specified attributes.
    """
    iou_data = {}
    #print(f"--- Processing {os.path.basename(filename)} ---") # DEBUG
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            #print(f"  File loaded. Top-level keys: {list(data.keys())}") # DEBUG

        for attribute in attributes_to_extract:
            #print(f"  Looking for top-level key: '{attribute}'") # DEBUG
            attr_data = data.get(attribute) # Directly get the attribute key

            if not attr_data or not isinstance(attr_data, dict):
                #print(f"    -> Attribute key '{attribute}' not found or not dict.") # DEBUG
                iou_data[attribute] = np.nan
                continue

            #print(f"    -> Found attribute '{attribute}'. Keys: {list(attr_data.keys())}") # DEBUG
            iou_dict = attr_data.get('IOU')

            if not iou_dict or not isinstance(iou_dict, dict):
                #print(f"    -> 'IOU' block not found or not dict for '{attribute}'.") # DEBUG
                iou_data[attribute] = np.nan
                continue

            #print(f"    -> Found 'IOU' block. Keys: {list(iou_dict.keys())}") # DEBUG
            mean_iou = iou_dict.get('mean')

            if mean_iou is None:
                #print(f"    -> 'mean' key not found in 'IOU' for '{attribute}'.") # DEBUG
                iou_data[attribute] = np.nan
            else:
                #print(f"    -> SUCCESS: Found mean={mean_iou} for '{attribute}'.") # DEBUG
                iou_data[attribute] = mean_iou

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filename}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred processing {filename}: {e}")
        return None

    # Return data ensuring all requested keys are present
    final_data = {key: iou_data.get(key, np.nan) for key in attributes_to_extract}
    #print(f"--- Finished {os.path.basename(filename)}. Returning data for {len(final_data)} attributes. ---") # DEBUG
    return final_data


# --- Load Data for all models ---
all_iou_data = {}

# Check file existence before processing
any_file_missing = False
for model_name, filename in file_mapping.items():
    if not os.path.exists(filename):
        print(f"Error: Input file not found: {filename}")
        any_file_missing = True
if any_file_missing:
    print("\nExiting because one or more input files were not found.")
    exit()

# Load data using the corrected function
for model_name, filename in file_mapping.items():
    print(f"\nExtracting data for {model_name} from {filename}...")
    # Use the FLAT extraction function and the flat attribute_order list
    iou_means = extract_iou_means_flat(filename, attribute_order)
    if iou_means is None:
        print(f"Exiting due to errors loading data for {model_name}.")
        exit()
    all_iou_data[model_name] = iou_means

# --- Prepare DataFrame ---
df = pd.DataFrame(all_iou_data)

# --- Order Data ---
# Reindex based on the desired attribute_order.
df_ordered = df.reindex(attribute_order)

# --- Data Cleaning for Plot ---
initial_rows = len(df_ordered)
df_plot = df_ordered.dropna(how='any')
final_rows = len(df_plot)

if final_rows == 0:
    print("\nError: No attributes found with valid IOU data across all three models after cleaning.")
    print("This likely means the extraction function failed to find 'IOU'->'mean' even with the corrected logic,")
    print("or the files have significantly different attribute sets / missing IOU blocks.")
    print("Please double-check:")
    print("  1. The exact file paths and names (especially 'gemma3_college_.json' for 4B).")
    print("  2. The JSON structure within EACH file *exactly* matches the provided samples (top-level attribute keys).")
    print("  3. The attribute keys in the JSON match the strings in attribute_order list (case-sensitive).")
    print("\nDataFrame before dropping NaNs (rows with any NaN were removed):")
    print(df_ordered.to_string())
    exit()
elif initial_rows > final_rows:
     dropped_attributes = list(set(df_ordered.index) - set(df_plot.index))
     print(f"\nNote: Dropped {initial_rows - final_rows} attributes due to missing IOU data in at least one model:")
     print(f" -> {dropped_attributes[:10]}{'...' if len(dropped_attributes) > 10 else ''}")


print(f"\nPlotting comparison for {final_rows} attributes with complete data across all models.")


# --- Create Visualization (Grouped Bar Chart) ---
plt.style.use('seaborn-v0_8-colorblind')
fig, ax = plt.subplots(figsize=(20, 10)) # Adjusted size

n_models = len(model_names)
n_attributes = len(df_plot)

bar_width = 0.25
index = np.arange(n_attributes)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, model_name in enumerate(model_names):
    bar_positions = index + (i - (n_models - 1) / 2) * bar_width
    ax.bar(bar_positions, df_plot[model_name], bar_width, label=model_name, color=colors[i % len(colors)], alpha=0.8)

ax.set_xlabel("Attribute", fontsize=12)
ax.set_ylabel("Mean IOU Score", fontsize=12)
ax.set_title("Comparison of Mean IOU Score Across Gemma Models by Attribute", fontsize=14, pad=20)

ax.set_xticks(index)
ax.set_xticklabels(df_plot.index, rotation=80, ha='right', fontsize=9)

ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(title="Model", fontsize=10, title_fontsize=11)

if not df_plot.empty:
    min_val = df_plot.min().min()
    max_val = df_plot.max().max()
    ax.set_ylim(bottom=max(0, min_val - (max_val-min_val)*0.05),
                top=max_val + (max_val-min_val)*0.05)
else:
    ax.set_ylim(bottom=0)

plt.tight_layout(pad=1.5)

# --- Save and Show Plot ---
os.makedirs(visualization_dir, exist_ok=True)
output_plot_file = os.path.join(visualization_dir, "iou_comparison_gemma_models_grouped.png")

try:
    plt.savefig(output_plot_file, dpi=300)
    print(f"\nPlot saved successfully to: {output_plot_file}")
except Exception as e:
    print(f"\nError saving plot: {e}")

plt.show()

print("\n--- Interpretation ---")
print("This grouped bar chart compares the average IOU scores for different Gemma models across various attributes.")
print("Each group represents an attribute; bars within a group show scores for Gemma 12B, 4B, and 1B.")
print("Compare bar heights within groups to see how IOU varies with model size for each attribute.")
print("Attributes missing data in *any* of the compared files are excluded from this plot.")