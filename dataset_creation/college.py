import kagglehub
import os
import pandas as pd
import json


DATASET_PATH = "data/datasets/"
ROOT_PATH = "../"

with open(f"{ROOT_PATH}.config", "r") as f:
    lines = f.readlines()
for line in lines:
    if line.startswith("KAGGLE_USERNAME"):
        username = line.split("=")[1].strip()
    elif line.startswith("KAGGLE_KEY"):
        key = line.split("=")[1].strip()

os.environ["KAGGLE_USERNAME"] = username
os.environ["KAGGLE_KEY"] = key


# Download latest version
path = kagglehub.dataset_download("tahirrfarooqq/2023-qs-world-university-ranking")
df = pd.read_csv(os.path.join(path, "2023 QS World University Rankings.csv"))

colleges = df.institution.values.tolist()
with open(f"{ROOT_PATH}{DATASET_PATH}college.json", "w") as f:
    json.dump(colleges, f, ensure_ascii=False, indent=4)
