import ollama
import json
from users import User

from utils.utils import extract_list_from_response, get_correct_file_name
from model_inf import LLMInterface, OllamaClient, VLLMClient
from tqdm import tqdm

import argparse
import configparser

parser = argparse.ArgumentParser()
parser.add_argument("--type_inf", type=str, default="ollama")
parser.add_argument("--models", type=str, default="llama3.2")
parser.add_argument("--dataset_types", type=str, default="college")
parser.add_argument("--type_of_activities", type=str, default="student")
parser.add_argument("--k", type=int)
parser.add_argument("--seeds", type=str, default=42)

args = parser.parse_args()

config = configparser.ConfigParser()
config.read("config/config_inference")


OUTPUT_PATH = config["paths"]["output_path"]
DATASET_PATH = config["paths"]["dataset_path"]
SENSITIVE_ATRIBUTES_PATH = config["paths"]["sensitive_attributes_path"]

models = config["parameters"]["models"].split(", ")
dataset_types = config["parameters"]["dataset_types"].split(", ")
type_of_activities = config["parameters"]["type_of_activities"].split(", ")
k = int(config["parameters"]["k"])
type_inf = config["parameters"]["type_inf"]
seeds = [int(seed) for seed in config["parameters"]["seeds"].split(", ")]
if type_of_activities[0] == "None":
    type_of_activities = ["","",""]

# if output_path direcorty does not exist, create it
import os
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    


def inference(model, dataset_type, k, type_of_activity, type_inf,seed,nb_calls,nb_errors):
    print(f"Running inference for {model} with {type_inf} on {dataset_type} as {type_of_activity} with seed {seed}")
    with open(f"{DATASET_PATH}{dataset_type}.json", "r") as f:
        items = json.load(f)
    original_items_set = set(items)

    options = {
        "seed": seed,
    }

    if type_inf == "ollama":
        llm_client = OllamaClient(options=options)
    elif type_inf == "vllm":
        llm_client = VLLMClient(options=options)

    neutral_user = User(
        dataset_type=dataset_type,
        items=items,
        k=k,
        type_of_activity=type_of_activity,
        sensitive_atribute="a",
    )

    prompts = neutral_user.build_prompts()
    response = llm_client.chat(model=model, messages=prompts)
    neutral_list, neutral_error_count = extract_list_from_response(
        response,
        original_items_set=original_items_set,
        k=k
    )
    nb_calls += 1
    nb_errors += neutral_error_count


    final_outputs = {
        "neutral": {
            "recommended_list": neutral_list,
            "response": response,
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
            response = llm_client.chat(model=model, messages=prompts)
            extracted_list, error_count = extract_list_from_response(
                response,
                original_items_set=original_items_set,
                k=k
            )
            nb_calls += 1
            nb_errors += error_count

            outputs[sensitive_atribute] = {
                "recommended_list": extracted_list,
                "response": response,
                "nb_items": len(extracted_list),
            }

        final_outputs[type_of_sensitive_atributes] = outputs

    name_file = get_correct_file_name(f"{model}_{dataset_type}_{type_of_activity}_{seed}.json")
    print(f"Saving {name_file}")
    file = f"{OUTPUT_PATH}{name_file}"
    # remove / in the file name
    with open(file, "w") as f:
        json.dump(final_outputs, f, indent=4)

    return nb_calls, nb_errors


if __name__ == "__main__":
    nb_calls = 0
    nb_errors = 0

    for model in models:
        for seed in seeds:
            for dataset_type, type_of_activity in zip(dataset_types, type_of_activities):
                nb_calls, nb_errors = inference(model, dataset_type, k, type_of_activity, type_inf,seed,nb_calls,nb_errors)

    print(f"Number of errors: {nb_errors} for {nb_calls} calls")
