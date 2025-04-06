import ollama
import json
from users import User

from utils import extract_list_from_response
from metrics import calc_iou, calc_serp_ms, calc_prag, get_item_rank
from tqdm import tqdm


# TODO need to create a config file for this
RESULT_PATH = "data/results/"
OUTPUT_PATH = "data/outputs/"
DATASET_PATH = "data/datasets/college.json"
SENSITIVE_ATRIBUTES_PATH = "data/sensitive_attributes.json"

model = "llama3.2"
dataset_type = "college"
k = 20
type_of_activity = "student"

with open(DATASET_PATH, "r") as f:
    items = json.load(f)

neutral_user = User(
    dataset_type=dataset_type,
    items=items,
    k=k,
    type_of_activity=type_of_activity,
    sensitive_atribute="a",
)
items_rank = {item: i for i, item in enumerate(items)}

prompts = neutral_user.build_prompts()
response = ollama.chat(model=model, messages=prompts)
neutral_list = extract_list_from_response(response)
mean_rank = get_item_rank(neutral_list, items_rank)


final_metrics = {}
final_outputs = {
    "neutral": {
        "recommended_list": neutral_list,
        "response": response["message"]["content"],
    }
}
with open(SENSITIVE_ATRIBUTES_PATH, "r") as f:
    dict_sensitive_atributes = json.load(f)

for type_of_sensitive_atributes in tqdm(
    dict_sensitive_atributes, desc="Processing all sensitive attributes", position=0
):
    sensitive_atributes = dict_sensitive_atributes[type_of_sensitive_atributes]
    metrics = {}
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
        if len(extracted_list) == 0:
            metrics[sensitive_atribute] = {}
            continue

        metrics[sensitive_atribute] = {
            "IOU": calc_iou(neutral_list, extracted_list),
            "SERP MS": calc_serp_ms(neutral_list, extracted_list),
            "Pragmatic": calc_prag(neutral_list, extracted_list),
            "mean_rank": get_item_rank(extracted_list, items_rank),
        }
        outputs[sensitive_atribute] = {
            "recommended_list": extracted_list,
            "response": response["message"]["content"],
        }

    final_metrics[type_of_sensitive_atributes] = metrics
    final_outputs[type_of_sensitive_atributes] = outputs

file = f"{RESULT_PATH}/{model}_{dataset_type}.json"
with open(file, "w") as f:
    json.dump(final_metrics, f, indent=4)

file = f"{OUTPUT_PATH}/{model}_{dataset_type}.json"
with open(file, "w") as f:
    json.dump(final_outputs, f, indent=4)
