import argparse
import configparser

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

from utils.utils import get_correct_file_name
import matplotlib

# matplotlib.use("Agg")  # Use a non-interactive backend for environments without display


DESIRED_ORDER = {
    "gender": ["a girl", "a boy", "a female", "a male"],
    "country": [
        "an American",
        "a Brazilian",
        "a British",
        "a Chinese",
        "a French",
        "a German",
        "a Japanese",
    ],
    "continent": ["an African", "an American", "an Asian"],
    "race": ["a Black", "a White", "an Asian", "a Hispanic"],
    "occupation": [
        "a child of a doctor",
        "a child of a student",
        "a child of a teacher",
        "a child of a worker",
        "a child of a writer",
    ],
    "religion": ["a Buddhist", "a Christian", "a Muslim"],
}

CATEGORY_ORDER = [
    "country",
    "gender",
    "continent",
    "occupation",
    "race",
    "religion",
    "physics",
]


parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str, default=["llama3.2"])
# parser.add_argument("--dataset_types", type=str, default=["college"])
parser.add_argument("--type_of_activities", type=str, default=["student"])
parser.add_argument("--k", type=int)
parser.add_argument("--seeds", type=str, default="0, 1")

config = configparser.ConfigParser()
config.read("config/config_inference")


OUTPUT_PATH = config["paths"]["output_path"]
RESULT_PATH = config["paths"]["result_path"]
DATASET_PATH = config["paths"]["dataset_path"]
VISUALIZATION_PATH = config["paths"]["visualization_path"]
SENSITIVE_ATRIBUTES_PATH = config["paths"]["sensitive_attributes_path"]

models = config["parameters"]["models"].split(", ")
dataset_types = ["movie", "movie"]
type_of_activities = ["", "action movie fan"]
models = config["parameters"]["models"].split(", ")

df_activities = {}
for model in models:
    for dataset_type, type_of_activity in zip(dataset_types, type_of_activities):
        name_save = get_correct_file_name(f"{model}_{dataset_type}_{type_of_activity}")
        filepath = f"{RESULT_PATH}{name_save}.json"

        with open(filepath, "r") as f:
            final_metrics = json.load(f)

        final_metrics.pop("neutral", None)

        plot_data_list = []
        original_attributes_present = list(final_metrics.keys())

        for attribute, metrics_dict in final_metrics.items():
            cleaned_attribute = attribute.replace("an ", "").replace("a ", "").title()
            for metric_name, stats_dict in metrics_dict.items():
                if metric_name == "mean_rank":
                    continue
                plot_data_list.append(
                    {
                        "Attribute": cleaned_attribute,
                        "Metric": metric_name,
                        "Mean": stats_dict["mean"],
                        "StdDev": stats_dict["std"],
                    }
                )

        df_plot = pd.DataFrame(plot_data_list)

        df_pivot_mean = df_plot.pivot(
            index="Attribute", columns="Metric", values="Mean"
        )
        df_pivot_std = df_plot.pivot(
            index="Attribute", columns="Metric", values="StdDev"
        )

        ordered_plot_attributes = []
        for category in CATEGORY_ORDER:
            if category in DESIRED_ORDER:
                for original_attribute in DESIRED_ORDER[category]:
                    if original_attribute in original_attributes_present:
                        cleaned_attribute_for_order = (
                            original_attribute.replace("an ", "")
                            .replace("a ", "")
                            .title()
                        )
                        if cleaned_attribute_for_order not in ordered_plot_attributes:
                            ordered_plot_attributes.append(cleaned_attribute_for_order)

        attributes_in_pivot = df_pivot_mean.index.tolist()
        final_ordered_attributes = [
            attr for attr in ordered_plot_attributes if attr in attributes_in_pivot
        ]

        df_pivot_mean = df_pivot_mean.reindex(final_ordered_attributes)
        df_pivot_std = df_pivot_std.reindex(final_ordered_attributes)

        df_activities[f"{type_of_activity}_mean"] = df_pivot_mean
        df_activities[f"{type_of_activity}_std"] = df_pivot_std

        sns.set_style("whitegrid")

        ax = df_pivot_mean.plot(
            kind="bar",
            yerr=df_pivot_std,
            figsize=(12, 6),
            capsize=3,
            colormap="tab10",
            width=0.8,
        )

        plt.title(f"Metric Scores by Attribute ({model} on {dataset_type})")
        plt.ylabel("Score")
        plt.xlabel("Attribute")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Metric")
        plt.tight_layout()


# Combine all DataFrames in df_activities into a single DataFrame
combined_df = pd.concat(df_activities, axis=0, names=["Activity", "Attribute"])

# Reset the index for better readability
combined_df.reset_index(inplace=True)

# Split the 'Activity' column into 'Activity Name' and 'Statistic' (mean/std)
combined_df[["Activity Name", "Statistic"]] = combined_df["Activity"].str.rsplit(
    "_", n=1, expand=True
)

# Pivot the DataFrame to have 'mean' and 'std' as separate columns
combined_df = combined_df.pivot(
    index=["Attribute", "Activity Name"], columns="Statistic", values="IOU Divergence"
).reset_index()

# Rename the columns for clarity
combined_df.columns.name = None  # Remove the columns' name
combined_df.rename(
    columns={"mean": "Mean IOU Divergence", "std": "Std IOU Divergence"}, inplace=True
)


category_to_attribute = {
    category: [attr.replace("an ", "").replace("a ", "").title() for attr in attributes]
    for category, attributes in DESIRED_ORDER.items()
}

# Create a spider plot for each category comparing Mean IOU Divergence by Activity Name
categories = category_to_attribute.keys()


sns.set_palette("muted")
sns.set_style("whitegrid")

# Create the plot
fig, axes = plt.subplots(2, 3, figsize=(20, 10), subplot_kw=dict(polar=True))
axes = axes.flatten()

for i, category in enumerate(categories):
    ax = axes[i]
    subset = combined_df[combined_df["Attribute"].isin(category_to_attribute[category])]
    attributes = subset["Attribute"].unique()

    if i == 5:
        angles = np.pi / 2 + np.linspace(0, 2 * np.pi, len(attributes), endpoint=False)
        print(f"Angles: {angles}")
    elif len(attributes) % 2 == 0:
        angles = np.pi / len(attributes) + np.linspace(
            0, 2 * np.pi, len(attributes), endpoint=False
        )
    else:
        angles = np.pi / (2 * len(attributes)) + np.linspace(
            0, 2 * np.pi, len(attributes), endpoint=False
        )
    angles = angles.tolist()

    ax.grid(True)
    ax.set_xticks(angles)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_rlabel_position(-10)

    # Iterate through activity names for each category
    for activity_name in subset["Activity Name"].unique():

        activity_data = subset[subset["Activity Name"] == activity_name]
        values = activity_data["Mean IOU Divergence"].tolist()
        values += values[:1]  # Close the circle

        label = activity_name
        if activity_name == "":
            label = "No Context"

        if i != 0:
            label = ""

        # Plot the data with custom line style and markers
        ax.plot(
            angles + [angles[0]],
            values,
            label=label,
            linewidth=2,
            marker="o",
            markersize=6,
        )
        ax.fill(angles + [angles[0]], values, alpha=0.3)  # Slightly transparent fill

    if category == "occupation":
        category = "Parent's occupation"
    # Set the title with improved styling
    if i >= len(axes) - 3:  # Bottom row
        ax.text(
            0.5,
            -0.2,  # x, y coordinates (0.5 center horizontally, -0.2 down below)
            category.capitalize(),
            transform=ax.transAxes,  # Important! tells it to use axes coordinates
            ha="center",
            va="center",
            fontsize=32,
            fontweight="bold",
        )
    else:  # Top row
        ax.text(
            0.5,
            1.15,  # x, y coordinates (0.5 center horizontally, -0.2 down below)
            category.capitalize(),
            transform=ax.transAxes,  # Important! tells it to use axes coordinates
            ha="center",
            va="center",
            fontsize=32,
            fontweight="bold",
        )

    attributes = [attr.replace("Child Of ", "") for attr in attributes]
    ax.set_xticklabels(attributes, fontsize=26, fontweight="light")

    # Remove y-ticks (no need for numerical values)
    ax.set_ylim(0, 1)

# Add a legend outside the plot with improved styling
fig.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=3,
    fontsize=28,
    frameon=True,
    shadow=True,
)

plt.subplots_adjust(hspace=0.5, wspace=-0.3, top=0.85, bottom=0.1)
plot_filename = f"{VISUALIZATION_PATH}{model}_movie_spider_plot_context_or_not.png"
plt.savefig(plot_filename, dpi=300)
print(f"Saved ordered plot to: {plot_filename}")
plt.show()  # Show the plot (or save it if needed)
plt.close()
