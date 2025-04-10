from utils.metrics import calc_iou, calc_serp_ms, calc_prag, get_item_rank

import argparse
import configparser
import json


import argparse
import configparser

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str, default=["llama3.2"])
parser.add_argument("--dataset_types", type=str, default=["college"])
parser.add_argument("--type_of_activities", type=str, default=["student"])
parser.add_argument("--k", type=int)


args = parser.parse_args()

config = configparser.ConfigParser()
config.read("config_inference")


OUTPUT_PATH = config["paths"]["output_path"]
RESULT_PATH = config["paths"]["result_path"]
DATASET_PATH = config["paths"]["dataset_path"]
SENSITIVE_ATRIBUTES_PATH = config["paths"]["sensitive_attributes_path"]

models = config["parameters"]["models"].split(", ")
dataset_types = config["parameters"]["dataset_types"].split(", ")
type_of_activities = config["parameters"]["type_of_activities"].split(", ")
k = int(config["parameters"]["k"])


def results(model, dataset_type, k, type_of_activity):
    print(f"Running results for {model} on {dataset_type} as {type_of_activity}")
    with open(f"{DATASET_PATH}{dataset_type}.json", "r") as f:
        items = json.load(f)
    items_rank = {item: i for i, item in enumerate(items)}

    file = f"{OUTPUT_PATH}{model}_{dataset_type}.json"
    with open(file, "r") as f:
        model_outputs = json.load(f)

    neutral_list = model_outputs["neutral"]["recommended_list"]
    neutral_rank = get_item_rank(neutral_list, items_rank)

    final_metrics = {
        "neutral": {
            "a neutral": {
                "mean_rank": neutral_rank,
            }
        }
    }
    del model_outputs["neutral"]

    for type_sensitive_atribute in model_outputs:
        for sensitive_atribute, user_outputs in model_outputs[
            type_sensitive_atribute
        ].items():
            if user_outputs["nb_items"] == 0:
                continue
            extracted_list = user_outputs["recommended_list"]

            metrics = {
                "IOU": calc_iou(neutral_list, extracted_list),
                "SERP MS": calc_serp_ms(neutral_list, extracted_list),
                "Pragmatic": calc_prag(neutral_list, extracted_list),
                "mean_rank": get_item_rank(extracted_list, items_rank),
            }
            final_metrics[sensitive_atribute] = metrics

    file = f"{RESULT_PATH}/{model}_{dataset_type}.json"
    with open(file, "w") as f:
        json.dump(final_metrics, f, indent=4)


if __name__ == "__main__":
    for model in models:
        for dataset_type, type_of_activity in zip(dataset_types, type_of_activities):
            results(model, dataset_type, k, type_of_activity)
