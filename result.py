from collections import defaultdict

import numpy as np
from utils.metrics import calc_iou, calc_serp_ms, calc_prag, get_item_rank, calc_diversity, calc_repetition_count
from utils.utils import get_correct_file_name

import argparse
import configparser
import json


import argparse
import configparser

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str, default=["llama3.2"])
parser.add_argument("--dataset_types", type=str, default=["college"])
parser.add_argument("--user_personas", type=str, default=["student"])
parser.add_argument("--k", type=int)
parser.add_argument("--seeds", type=str, default="0, 1")


args = parser.parse_args()

config = configparser.ConfigParser()
config.read("config/config_inference")


OUTPUT_PATH = config["paths"]["output_path"]
RESULT_PATH = config["paths"]["result_path"]
DATASET_PATH = config["paths"]["dataset_path"]
SENSITIVE_ATRIBUTES_PATH = config["paths"]["sensitive_attributes_path"]

models = config["parameters"]["models"].split(", ")
dataset_types = config["parameters"]["dataset_types"].split(", ")
user_personas = config["parameters"]["user_personas"].split(", ")
k = int(config["parameters"]["k"])
seeds = [int(seed) for seed in config["parameters"]["seeds"].split(", ")]
if user_personas[0] == "None":
    user_personas = ["","",""]

# if result_path directory does not exist, create it
import os
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

def results(model, dataset_type, k, user_persona,seed):
    print(f"Running results for {model} on {dataset_type} as {user_persona}")
    with open(f"{DATASET_PATH}{dataset_type}.json", "r") as f:
        items = json.load(f)
    items_rank = {item: i for i, item in enumerate(items)}

    file_name = get_correct_file_name(f"{model}_{dataset_type}_{user_persona}_{seed}.json")
    with open(f"{OUTPUT_PATH}{file_name}", "r") as f:
        model_outputs = json.load(f)

    neutral_list = model_outputs["neutral"]["recommended_list"]
    neutral_rank = get_item_rank(neutral_list, items_rank)
    neutral_diversity = calc_diversity(neutral_list)

    final_metrics = {
        "neutral": {
                "mean_rank": neutral_rank,
                "diversity": neutral_diversity,
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
                "IOU Divergence": 1-calc_iou(neutral_list, extracted_list),
                "SERP MS Divergence": 1-calc_serp_ms(neutral_list, extracted_list),
                "Pragmatic Divergence": 1-calc_prag(neutral_list, extracted_list),
                "mean_rank" : get_item_rank(extracted_list, items_rank),
                "diversity": calc_diversity(extracted_list),
            }
            final_metrics[sensitive_atribute] = metrics

    return final_metrics

    # file = f"{RESULT_PATH}{model}_{dataset_type}.json"
    # with open(file, "w") as f:
    #     json.dump(final_metrics, f, indent=4)

def aggregate_and_save_metrics(metrics_by_seed, model, dataset_type, user_persona):
    print(f"Processing results for {model}, Dataset: {dataset_type}")
    # neutral rank
    collected_values = defaultdict(lambda: defaultdict(list))

    for seed, seed_metrics in metrics_by_seed.items():
        for attribute, metrics_or_nested in seed_metrics.items():
            for metric_name, value in metrics_or_nested.items():
                collected_values[attribute][metric_name].append(value)
    
    aggregated_output = defaultdict(lambda: defaultdict(dict))
    for attribute, metrics_dict in collected_values.items():
        for metric_name, values_list in metrics_dict.items():
            np_values = np.array(values_list)
            mean_val = np.mean(np_values)
            std_val = np.std(np_values)
            num_aggregated = len(values_list)

            aggregated_output[attribute][metric_name] = {
                "mean": mean_val,
                "std": std_val,
                "num_values": num_aggregated
            }
    file_name = get_correct_file_name(f"{model}_{dataset_type}_{user_persona}.json")
    file = f"{RESULT_PATH}{file_name}"
    with open(file, "w") as f:
        json.dump(aggregated_output, f, indent=4)

if __name__ == "__main__":
    for model in models:
        for dataset_type, user_persona in zip(dataset_types, user_personas):
            metric_for_seed_aggragation = {}
            for seed in seeds:
                metric_for_seed_aggragation[seed] = results(model, dataset_type, k, user_persona,seed)
            # save aggregated results
            print(metric_for_seed_aggragation)
            aggregate_and_save_metrics(
                metric_for_seed_aggragation,
                model,
                dataset_type,
                user_persona,
            )


