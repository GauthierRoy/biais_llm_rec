import argparse
import configparser

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import json
import numpy as np

from utils.utils import get_correct_file_name


parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str, default=["llama3.2"])
# parser.add_argument("--dataset_types", type=str, default=["college"])
parser.add_argument("--type_of_activities", type=str, default=["student"])
parser.add_argument("--k", type=int)
parser.add_argument("--seeds", type=str, default="0, 1")

config = configparser.ConfigParser()
config.read("config_inference")


OUTPUT_PATH = config["paths"]["output_path"]
RESULT_PATH = config["paths"]["result_path"]
DATASET_PATH = config["paths"]["dataset_path"]
VISUALIZATION_PATH = config["paths"]["visualization_path"]
SENSITIVE_ATRIBUTES_PATH = config["paths"]["sensitive_attributes_path"]

models = config["parameters"]["models"].split(", ")
dataset_types = config["parameters"]["dataset_types"].split(", ")
type_of_activities = config["parameters"]["type_of_activities"].split(", ")
models = config["parameters"]["models"].split(", ")

for model in models:
    for (dataset_type, type_of_activity) in zip(dataset_types, type_of_activities):
        name_save = get_correct_file_name(f"{model}_{dataset_type}_{type_of_activity}")
        filepath = f"{RESULT_PATH}{name_save}.json"

        with open(filepath, "r") as f:
            final_metrics = json.load(f)

        final_metrics.pop("neutral", None)

        plot_data_list = []
        for attribute, metrics_dict in final_metrics.items():
            cleaned_attribute = attribute.replace("an ", "").replace("a ", "").title()
            for metric_name, stats_dict in metrics_dict.items():
                if metric_name == "mean_rank":
                    continue
                plot_data_list.append({
                    "Attribute": cleaned_attribute,
                    "Metric": metric_name,
                    "Mean": stats_dict["mean"],
                    "StdDev": stats_dict["std"]
                })

        df_plot = pd.DataFrame(plot_data_list)

        # --- Pivot Data ---
        df_pivot_mean = df_plot.pivot(index="Attribute", columns="Metric", values="Mean")
        df_pivot_std = df_plot.pivot(index="Attribute", columns="Metric", values="StdDev")

        # --- Plotting ---
        sns.set_style("whitegrid")

        ax = df_pivot_mean.plot(
            kind='bar',
            yerr=df_pivot_std,
            figsize=(12, 6),
            capsize=3,
            colormap='tab10', # <--- FIXED: Use a valid Matplotlib colormap
            width=0.8
        )

        # Basic Plot Customization
        plt.title(f"Metric Scores by Attribute ({model} on {dataset_type})")
        plt.ylabel("Score")
        plt.xlabel("Attribute")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Metric")
        plt.tight_layout()
        plt.show()

        # Save the plot
        plt.savefig(f"{VISUALIZATION_PATH}{name_save}_plot.png", dpi=300)
        plt.close()  # Close the plot to free memory


        # --- Create LaTeX Table ---


        table_data_list = []
        for attribute, metrics_dict in final_metrics.items():
            # Clean the attribute name for the 'Group' column
            cleaned_group_name = attribute.replace("an ", "").replace("a ", "").title()

            # Get the mean rank stats, safely handle if 'mean_rank' or 'mean' is missing
            mean_rank_stats = metrics_dict.get("mean_rank", {}) # Get stats dict or empty dict
            mean_rank_value = mean_rank_stats.get("mean", np.nan) # Get mean or NaN

            table_data_list.append({
                "Attribute": attribute, # Keep original attribute if needed
                "Group": cleaned_group_name,
                "Mean Rank": mean_rank_value
            })

        # Create the DataFrame
        df_merged = pd.DataFrame(table_data_list)

        # Optional: Sort the DataFrame if desired (e.g., by Group)
        # df_merged = df_merged.sort_values(by="Group")

        # --- Generate LaTeX Table ---
        # Use float_format for nice rounding in the output
        latex_table = df_merged.to_latex(
            index=False,
            caption="Mean Rank by Attribute Group",
            label="tab:mean_rank",
            na_rep='NaN', # How to represent NaN values in LaTeX
            float_format="%.2f" # Format floats to 2 decimal places
        )

        print(latex_table)
        # save it in a file

        with open(f"{VISUALIZATION_PATH}{name_save}_latex_table.tex", "w") as f:
            f.write(latex_table)

        for key, value in final_metrics.items():
            print(f"{key}: {value}")

            df = pd.DataFrame(value)

            df_reset = df.reset_index().rename(columns={"index": "Metric"})
            df_long = df_reset.melt(id_vars="Metric", var_name=key, value_name="Value")

            df_long[key] = df_long[key].str.replace("an |a ", "", regex=True).str.title()
            df_long = df_long[df_long["Metric"] != "mean_rank"]

            df_long

            break

        final_metrics