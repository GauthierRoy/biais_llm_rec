import json
import os
import configparser
import gc
import torch
from tqdm import tqdm

# Imports
from users import User
from utils.utils import extract_list_from_response, get_correct_file_name
from model_inf import VLLMOfflineClient

# --- Load Config ---
config = configparser.ConfigParser()
config.read("config/config_inference")

OUTPUT_PATH = config["paths"]["output_path"]
DATASET_PATH = config["paths"]["dataset_path"]
SENSITIVE_ATRIBUTES_PATH = config["paths"]["sensitive_attributes_path"]

models = config["parameters"]["models"].split(", ")
dataset_types = config["parameters"]["dataset_types"].split(", ")
user_personas = config["parameters"]["user_personas"].split(", ")
k = int(config["parameters"]["k"])
seeds = [int(seed) for seed in config["parameters"]["seeds"].split(", ")]

if user_personas[0] == "None":
    user_personas = ["", "", ""]

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


def inference_batched(client: VLLMOfflineClient, model_name, dataset_type, k, user_persona, seed, nb_calls, nb_errors):
    
    # 1. Load Data
    with open(f"{DATASET_PATH}{dataset_type}.json", "r") as f:
        items = json.load(f)
    original_items_set = set(items)

    # 2. Build Request Batch
    requests_batch = [] 

    # -- Neutral User --
    neutral_user = User(
        dataset_type=dataset_type, items=items, k=k,
        user_persona=user_persona, sensitive_atribute="a"
    )
    requests_batch.append({
        "type": "neutral",
        "attr": "neutral",
        "prompts": neutral_user.build_prompts()
    })

    # -- Sensitive Users --
    with open(SENSITIVE_ATRIBUTES_PATH, "r") as f:
        dict_sensitive_atributes = json.load(f)

    for type_attr, attr_list in dict_sensitive_atributes.items():
        for sensitive_atribute in attr_list:
            user = User(
                dataset_type=dataset_type, items=items, k=k,
                user_persona=user_persona, sensitive_atribute=sensitive_atribute
            )
            requests_batch.append({
                "type": "sensitive",
                "group": type_attr,
                "attr": sensitive_atribute,
                "prompts": user.build_prompts()
            })

    # 3. Execute Batch Inference
    # Extract only the messages
    all_messages_list = [req["prompts"] for req in requests_batch]
    
    print(f"    -> [GPU] Generating {len(all_messages_list)} responses...")
    
    # vLLM will now show its own progress bar here because we set use_tqdm=True in model_inf.py
    all_responses = client.chat_batch(all_messages_list)

    # 4. Process Results & Logging
    final_outputs = {"neutral": {}}
    error_log_path = "errors.txt"

    # Clean TQDM Loop for CPU work
    iterator = zip(requests_batch, all_responses)
    
    # Renamed description to "Parsing Results"
    with tqdm(iterator, total=len(requests_batch), desc=f"    -> Parsing {dataset_type}", leave=False) as pbar:
        for req, response_text in pbar:
            
            # Silent Extraction
            extracted_list, error_count, invalid_items = extract_list_from_response(
                response_text, 
                original_items_set=original_items_set, 
                k=k
            )
            
            nb_calls += 1
            nb_errors += error_count
            
            # Dynamic Bar Update
            if error_count > 0:
                pbar.set_postfix({"Err": nb_errors, "Last": f"{error_count} bad"})
                
                # Log to file
                with open(error_log_path, "a") as f:
                    f.write(f"\n[Model: {model_name} | Seed: {seed} | Group: {req.get('group', 'neutral')}]\n")
                    for bad in invalid_items:
                        f.write(f" -> {bad}\n")
            else:
                 pbar.set_postfix({"Err": nb_errors})

            # Structuring Output
            result_data = {
                "recommended_list": extracted_list,
                "response": response_text,
                "nb_items": len(extracted_list),
            }

            if req["type"] == "neutral":
                final_outputs["neutral"] = result_data
            else:
                group = req["group"]
                attr = req["attr"]
                if group not in final_outputs:
                    final_outputs[group] = {}
                final_outputs[group][attr] = result_data

    # 5. Save JSON
    name_file = get_correct_file_name(f"{model_name}_{dataset_type}_{user_persona}_{seed}.json")
    with open(f"{OUTPUT_PATH}{name_file}", "w") as f:
        json.dump(final_outputs, f, indent=4)

    return nb_calls, nb_errors


# --- Main Entry Point ---
if __name__ == "__main__":
    nb_calls = 0
    nb_errors = 0

    print("=== Starting Offline Batch Inference Pipeline ===")

    for model in models:
        print(f"\n>>> Initializing Model: {model} <<<")
        
        try:
            llm_client = VLLMOfflineClient(
                model_key=model, 
                options={"temperature": 0.7, "max_tokens": 1024}
            )
        except Exception as e:
            print(f"CRITICAL ERROR loading {model}: {e}")
            continue

        # Position=0 ensures this bar stays at the top
        for seed in tqdm(seeds, desc=f"Seeds ({model})", position=0):
            llm_client.sampling_params.seed = seed
            
            for dataset_type, user_persona in zip(dataset_types, user_personas):
                nb_calls, nb_errors = inference_batched(
                    client=llm_client,
                    model_name=model,
                    dataset_type=dataset_type,
                    k=k,
                    user_persona=user_persona,
                    seed=seed,
                    nb_calls=nb_calls,
                    nb_errors=nb_errors
                )

        print(f">>> Unloading {model}...")
        del llm_client
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nAll Done. Total Errors: {nb_errors} / Total Calls: {nb_calls}")