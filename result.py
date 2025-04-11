from collections import defaultdict

import numpy as np
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
parser.add_argument("--seeds", type=str, default="0, 1")


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
seeds = [int(seed) for seed in config["parameters"]["seeds"].split(", ")]


def results(model, dataset_type, k, type_of_activity,seed):
    print(f"Running results for {model} on {dataset_type} as {type_of_activity}")
    with open(f"{DATASET_PATH}{dataset_type}.json", "r") as f:
        items = json.load(f)
    items_rank = {item: i for i, item in enumerate(items)}

    file = f"{OUTPUT_PATH}{model}_{dataset_type}_{seed}.json"
    with open(file, "r") as f:
        model_outputs = json.load(f)

    neutral_list = model_outputs["neutral"]["recommended_list"]
    neutral_rank = get_item_rank(neutral_list, items_rank)

    final_metrics = {
        "neutral": {
                "mean_rank": neutral_rank,
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

    return final_metrics

    # file = f"{RESULT_PATH}{model}_{dataset_type}.json"
    # with open(file, "w") as f:
    #     json.dump(final_metrics, f, indent=4)

def aggregate_and_save_metrics(metrics_by_seed, model, dataset_type, output_base_path, processed_seeds):
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
    file = f"{RESULT_PATH}{model}_{dataset_type}.json"
    with open(file, "w") as f:
        json.dump(aggregated_output, f, indent=4)

if __name__ == "__main__":
    for model in models:
        metric_for_seed_aggragation = {}
        for seed in seeds:
            for dataset_type, type_of_activity in zip(dataset_types, type_of_activities):
                metric_for_seed_aggragation[seed] = results(model, dataset_type, k, type_of_activity,seed)
        # save aggregated results
        print(metric_for_seed_aggragation)
        aggregate_and_save_metrics(
            metric_for_seed_aggragation,
            model,
            dataset_type,
            OUTPUT_PATH,
            seeds
        )


