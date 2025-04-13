import argparse
import configparser

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

from utils.utils import get_correct_file_name


DESIRED_ORDER = {
    "country": [
        "an American",
        "a Brazilian",
        "a British",
        "a Chinese",
        "a French",
        "a German",
        "a Japanese"
    ],
    "gender": [
        "a girl",
        "a boy",
        "a female",
        "a male"
    ],
    "continent": [
        "an African",
        "an American",
        "an Asian"
    ],
    "occupation": [
        "a child of a doctor",
        "a child of a student",
        "a child of a teacher",
        "a child of a worker",
        "a child of a writer"
    ],
    "race": [
        "a Black",
        "a White",
        "an Asian",
        "a Hispanic"
    ],
    "religion": [
        "a Buddhist",
        "a Christian",
        "a Muslim"
    ],
    "physics": [
        "fat",
        "thin"
    ]
}

CATEGORY_ORDER = [
    "country",
    "gender",
    "continent",
    "occupation",
    "race",
    "religion",
    "physics"
]


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

if type_of_activities[0] == "None":
    type_of_activities = ["","",""]

for model in models:
    for (dataset_type, type_of_activity) in zip(dataset_types, type_of_activities):
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
                plot_data_list.append({
                    "Attribute": cleaned_attribute,
                    "Metric": metric_name,
                    "Mean": stats_dict["mean"],
                    "StdDev": stats_dict["std"]
                })

        df_plot = pd.DataFrame(plot_data_list)

        df_pivot_mean = df_plot.pivot(index="Attribute", columns="Metric", values="Mean")
        df_pivot_std = df_plot.pivot(index="Attribute", columns="Metric", values="StdDev")

        ordered_plot_attributes = []
        for category in CATEGORY_ORDER:
            if category in DESIRED_ORDER:
                for original_attribute in DESIRED_ORDER[category]:
                    if original_attribute in original_attributes_present:
                        cleaned_attribute_for_order = original_attribute.replace("an ", "").replace("a ", "").title()
                        if cleaned_attribute_for_order not in ordered_plot_attributes:
                             ordered_plot_attributes.append(cleaned_attribute_for_order)

        attributes_in_pivot = df_pivot_mean.index.tolist()
        final_ordered_attributes = [attr for attr in ordered_plot_attributes if attr in attributes_in_pivot]

        df_pivot_mean = df_pivot_mean.reindex(final_ordered_attributes)
        df_pivot_std = df_pivot_std.reindex(final_ordered_attributes)


        sns.set_style("whitegrid")

        ax = df_pivot_mean.plot(
            kind='bar',
            yerr=df_pivot_std,
            figsize=(12, 6),
            capsize=3,
            colormap='tab10',
            width=0.8
        )

        plt.title(f"Metric Scores by Attribute ({model} on {dataset_type})")
        plt.ylabel("Score")
        plt.xlabel("Attribute")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Metric")
        plt.tight_layout()

        plot_filename = f"{VISUALIZATION_PATH}{name_save}_plot_ordered.png"
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved ordered plot to: {plot_filename}")
        plt.close()


        table_data_list = []
        for attribute, metrics_dict in final_metrics.items():
            cleaned_group_name = attribute.replace("an ", "").replace("a ", "").title()
            mean_rank_stats = metrics_dict.get("mean_rank", {})
            mean_rank_value = mean_rank_stats.get("mean", np.nan)

            table_data_list.append({
                "Attribute": attribute,
                "Group": cleaned_group_name,
                "Mean Rank": mean_rank_value
            })

        df_merged = pd.DataFrame(table_data_list)

        # df_merged = df_merged.sort_values(by="Group")

        latex_table = df_merged.to_latex(
            index=False,
            caption="Mean Rank by Attribute Group",
            label="tab:mean_rank",
            na_rep='NaN',
            float_format="%.2f"
        )

        print(latex_table)
        latex_filename = f"{VISUALIZATION_PATH}{name_save}_latex_table.tex"
        with open(latex_filename, "w") as f:
            f.write(latex_table)
        print(f"Saved LaTeX table to: {latex_filename}")


        for key, value in final_metrics.items():
            print(f"{key}: {value}")

            df = pd.DataFrame(value)

            df_reset = df.reset_index().rename(columns={"index": "Metric"})
            df_long = df_reset.melt(id_vars="Metric", var_name=key, value_name="Value")

            df_long[key] = df_long[key].str.replace("an |a ", "", regex=True).str.title()
            df_long = df_long[df_long["Metric"] != "mean_rank"]

            break

        # final_metrics