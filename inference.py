import ollama
import json
from users import User

from utils.utils import extract_list_from_response
from utils.metrics import calc_iou, calc_serp_ms, calc_prag, get_item_rank
from tqdm import tqdm

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
DATASET_PATH = config["paths"]["dataset_path"]
SENSITIVE_ATRIBUTES_PATH = config["paths"]["sensitive_attributes_path"]

models = config["parameters"]["models"].split(", ")
dataset_types = config["parameters"]["dataset_types"].split(", ")
type_of_activities = config["parameters"]["type_of_activities"].split(", ")
k = int(config["parameters"]["k"])


def inference(model, dataset_type, k, type_of_activity):
    print(f"Running inference for {model} on {dataset_type} as {type_of_activity}")
    with open(f"{DATASET_PATH}{dataset_type}.json", "r") as f:
        items = json.load(f)

    neutral_user = User(
        dataset_type=dataset_type,
        items=items,
        k=k,
        type_of_activity=type_of_activity,
        sensitive_atribute="a",
    )

    prompts = neutral_user.build_prompts()
    response = ollama.chat(model=model, messages=prompts)
    neutral_list = extract_list_from_response(response)

    final_outputs = {
        "neutral": {
            "recommended_list": neutral_list,
            "response": response["message"]["content"],
            "nb_items": len(neutral_list),
        }
    }

    with open(SENSITIVE_ATRIBUTES_PATH, "r") as f:
        dict_sensitive_atributes = json.load(f)

    for type_of_sensitive_atributes in tqdm(
        dict_sensitive_atributes, desc="Processing all sensitive attributes", position=0
    ):
        sensitive_atributes = dict_sensitive_atributes[type_of_sensitive_atributes]
        outputs = {}
        for sensitive_atribute in tqdm(
            sensitive_atributes,
            desc=f"Processing {type_of_sensitive_atributes} atributes",
            leave=False,
            position=1,
        ):
            user = User(
                dataset_type=dataset_type,
                items=items,
                k=k,
                type_of_activity=type_of_activity,
                sensitive_atribute=sensitive_atribute,
            )
            prompts = user.build_prompts()
            response = ollama.chat(model=model, messages=prompts)
            extracted_list = extract_list_from_response(response)

            outputs[sensitive_atribute] = {
                "recommended_list": extracted_list,
                "response": response["message"]["content"],
                "nb_items": len(extracted_list),
            }

        final_outputs[type_of_sensitive_atributes] = outputs

    file = f"{OUTPUT_PATH}/{model}_{dataset_type}.json"
    with open(file, "w") as f:
        json.dump(final_outputs, f, indent=4)


if __name__ == "__main__":
    for model in models:
        for dataset_type, type_of_activity in zip(dataset_types, type_of_activities):
            inference(model, dataset_type, k, type_of_activity)
