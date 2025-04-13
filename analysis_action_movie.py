import configparser
import json
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import find_best_match # Assuming find_best_match is in utils/utils.py

# --- Configuration ---
config = configparser.ConfigParser()
# Make sure 'config_inference' exists in the same directory or provide the full path
config_file_path = "config_inference"
if not os.path.exists(config_file_path):
    print(f"Error: Configuration file '{config_file_path}' not found.")
    exit()
config.read(config_file_path)

OUTPUT_PATH = config["paths"]["output_path"]
VISUALIZATION_PATH = config["paths"]["visualization_path"]

file_mapping = {
    # Using shorter names for legend brevity
    "Gemma 12B": 'gemma3_12b_college__3.json',
    "Gemma 4B": 'gemma3_4b_college__3.json',
    "Gemma 1B": 'gemma3_1b_college__3.json'
}
model_names = list(file_mapping.keys()) # ["Gemma 12B", "Gemma 4B", "Gemma 1B"]
visualization_dir = VISUALIZATION_PATH
data_dir = OUTPUT_PATH

# --- Desired Attribute Order and Category Mapping ---
attribute_order_nested = {
    "country": ["an American", "a Brazilian", "a British", "a Chinese", "a French", "a German", "a Japanese"],
    "gender": ["a girl", "a boy", "a female", "a male"],
    "continent": ["an African", "an American", "an Asian"],
    "occupation": ["a child of a doctor", "a child of a student", "a child of a teacher", "a child of a worker", "a child of a writer"],
    "race": ["a Black", "a White", "an Asian", "a Hispanic"],
    "religion": ["a Buddhist", "a Christian", "a Muslim"],
    "physics": ["fat", "thin"]
}

# Flatten the order list
attribute_order = []
# Create a map to quickly find the category of an attribute
attribute_to_category_map = {}
category_order = ["country", "gender", "continent", "occupation", "race", "religion", "physics"] # Define category order

for category in category_order:
    if category in attribute_order_nested:
        attributes_in_category = attribute_order_nested[category]
        attribute_order.extend(attributes_in_category)
        for attribute in attributes_in_category:
            attribute_to_category_map[attribute] = category
    else:
        print(f"Warning: Category '{category}' from category_order not found in attribute_order_nested.")

# --- Revised Helper Function to Extract IOU Means ---
def extract_iou_means_nested(filename, attr_to_cat_map):
    """
    Loads JSON data structured with categories (like 'gender', 'race')
    and extracts the mean IOU for specified attributes within their categories.
    """
    iou_data = {}
    processed_attributes = set() # Keep track of attributes we've tried to process

    try:
        with open(data_dir+filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for attribute, category in attr_to_cat_map.items():
            processed_attributes.add(attribute)
            # Get the category block (e.g., data['gender'])
            category_data = data.get(category)

            if not category_data or not isinstance(category_data, dict):
                # Special case for 'neutral' if it's top-level and needed (though not in order list)
                if attribute == "neutral" and "neutral" in data:
                     attr_data = data.get("neutral")
                     # Check for IOU->mean within neutral
                     iou_dict = attr_data.get('IOU', {}).get('mean') if isinstance(attr_data, dict) else None
                     if iou_dict is not None:
                         iou_data[attribute] = iou_dict
                     else:
                        # print(f"Debug: No IOU->mean for top-level 'neutral' in {filename}")
                        iou_data[attribute] = np.nan
                else:
                    # This category (e.g., 'occupation') is missing entirely
                    print(f"Warning: Category '{category}' not found or not a dict in {filename}. Cannot get attribute '{attribute}'.")
                    iou_data[attribute] = np.nan
                continue # Skip to next attribute if category is missing

            # Get the specific attribute data within the category (e.g., data['gender']['a girl'])
            attr_data = category_data.get(attribute)

            if attr_data and isinstance(attr_data, dict):
                iou_dict = attr_data.get('IOU')
                if iou_dict and isinstance(iou_dict, dict):
                    mean_iou = iou_dict.get('mean')
                    if mean_iou is not None:
                        iou_data[attribute] = mean_iou
                    else:
                        print(f"Warning: 'mean' key missing in 'IOU' for attribute '{attribute}' (Category: {category}) in {filename}")
                        iou_data[attribute] = np.nan
                else:
                    # Attribute block exists, but no IOU block
                    print(f"Warning: 'IOU' data missing/not dict for attribute '{attribute}' (Category: {category}) in {filename}")
                    iou_data[attribute] = np.nan
            else:
                # The specific attribute (e.g., 'a child of a doctor') is missing within its category ('occupation')
                print(f"Warning: Attribute '{attribute}' data not found/not dict within category '{category}' in {filename}")
                iou_data[attribute] = np.nan

        # Check if any attributes from the map were NOT processed (shouldn't happen with this logic)
        # for attr in attr_to_cat_map:
        #      if attr not in processed_attributes:
        #           print(f"Internal check: Attribute '{attr}' was unexpectedly not processed for {filename}")
        #           iou_data[attr] = np.nan


    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filename}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred processing {filename}: {e}")
        return None

    # Return data only for the keys requested in the map
    return {key: iou_data.get(key, np.nan) for key in attr_to_cat_map}


# --- Load Data for all models ---
all_iou_data = {}

for model_name, filename in file_mapping.items():
    print(f"\nExtracting data for {model_name} from {filename}...")
    # Use the revised function and the attribute-to-category map
    iou_means = extract_iou_means_nested(filename, attribute_to_category_map)
    if iou_means is None:
        print(f"Exiting due to errors loading data for {model_name}.")
        exit()
    all_iou_data[model_name] = iou_means

# --- Prepare DataFrame ---
df = pd.DataFrame(all_iou_data)

# --- Order Data ---
# Reindex based on the desired attribute_order. This automatically handles missing attributes from the files.
df_ordered = df.reindex(attribute_order)

# --- Data Cleaning for Plot ---
# Drop rows where *any* model has missing data (NaN) for a fair comparison
df_plot = df_ordered.dropna()

if df_plot.empty:
    print("\nError: No attributes found with valid IOU data across all three models after cleaning.")
    print("Please check the JSON files content and the attribute order definition.")
    # Optional: print df_ordered to see where NaNs occurred
    # print("\nDataFrame before dropping NaNs:")
    # print(df_ordered)
    exit()

print(f"\nPlotting comparison for {len(df_plot)} attributes with complete data across all models.")
# print(f"Attributes being plotted: {df_plot.index.tolist()}") # uncomment to see which attributes are plotted


# --- Create Visualization (Grouped Bar Chart) ---
plt.style.use('seaborn-v0_8-colorblind')
# Increased figure width further for potentially many attributes
fig, ax = plt.subplots(figsize=(20, 10))

n_models = len(model_names)
n_attributes = len(df_plot)

bar_width = 0.25 # Width of individual bars
index = np.arange(n_attributes) # Base positions for attribute groups

# Define distinct colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'] # Added more colors just in case

# Plot bars for each model
for i, model_name in enumerate(model_names):
    # Calculate center position for this model's bars within the group
    # Offset = (index within group) - (half number of models) + (half bar width)
    bar_positions = index + (i - (n_models - 1) / 2) * bar_width
    ax.bar(bar_positions, df_plot[model_name], bar_width, label=model_name, color=colors[i % len(colors)], alpha=0.8)

# Labels and Title
ax.set_xlabel("Attribute", fontsize=12)
ax.set_ylabel("Mean IOU Score", fontsize=12)
ax.set_title("Comparison of Mean IOU Score Across Gemma Models by Attribute", fontsize=14, pad=20)

# Customize x-axis ticks
ax.set_xticks(index) # Set ticks at the center of each group
ax.set_xticklabels(df_plot.index, rotation=80, ha='right', fontsize=9) # Increased rotation, smaller font

# Add grid lines for the y-axis
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Add legend
ax.legend(title="Model", fontsize=10, title_fontsize=11)

plt.ylim(bottom=0) # Start y-axis at 0 for IOU
plt.tight_layout(pad=1.5) # Adjust padding

# --- Save and Show Plot ---
os.makedirs(visualization_dir, exist_ok=True)
output_plot_file = os.path.join(visualization_dir, "iou_comparison_gemma_models_grouped.png")

try:
    plt.savefig(output_plot_file, dpi=300) # Removed bbox_inches='tight' temporarily to see if it helps label cutoff with tight_layout
    print(f"\nPlot saved successfully to: {output_plot_file}")
except Exception as e:
    print(f"\nError saving plot: {e}")

plt.show()

print("\n--- Interpretation ---")
print("This grouped bar chart compares the average IOU scores for different Gemma models across various attributes.")
print("Each group represents an attribute; bars within a group show scores for Gemma 12B, 4B, and 1B.")
print("Compare bar heights within groups to see how IOU varies with model size for each attribute.")
print("Attributes missing data in *any* of the compared files are excluded from this plot.")