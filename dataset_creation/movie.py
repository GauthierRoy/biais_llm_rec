import requests
import json
from tqdm import tqdm
import os

ROOT_PATH = "../"
URL = "https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&release_date.lte=01%2F01%2F2024&sort_by=popularity.desc&vote_average.gte=8"
DATASET_PATH = "data/datasets/"
TOP_K = 500

#  API call made the 22 April 2025


def get_one_page(page, url):
    url = f"{url}&page={page}"
    response = requests.get(url, headers=headers)
    return response.json()


def get_all_pages(total_pages, url, max_movies=None):
    all_data = []
    for page in tqdm(range(1, total_pages + 1)):
        if len(all_data) >= max_movies:
            break
        data = get_one_page(page, url)
        for movie in data["results"]:
            all_data.append(movie["title"])

    # keep the TOP_K movies
    if max_movies is not None:
        all_data = all_data[:max_movies]

    return all_data


with open(f"{ROOT_PATH}.config", "r") as f:
    lines = f.readlines()
for line in lines:
    if line.startswith("TMBD_BEARER_TOKEN"):
        token = line.split("=")[1].strip()


headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {token}",
}

response = requests.get(URL, headers=headers)

data = response.json()
total_pages = data["total_pages"]


movies = get_all_pages(total_pages, URL, max_movies=TOP_K)
with open(f"{ROOT_PATH}{DATASET_PATH}movie.json", "w") as f:
    json.dump(movies, f, ensure_ascii=False, indent=4)
