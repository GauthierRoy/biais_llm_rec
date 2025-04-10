import requests
import json
from tqdm import tqdm
import os

ROOT_PATH = "../"
URL = "https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&release_date.lte=01%2F01%2F2024&sort_by=popularity.desc&vote_average.gte=8"
DATASET_PATH = "data/datasets/"


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


def get_one_page(page, url):
    url = f"{url}&page={page}"
    response = requests.get(url, headers=headers)
    return response.json()


def get_all_pages(total_pages, url, max_pages=None):
    all_data = []
    for page in tqdm(range(1, total_pages + 1)):
        if max_pages and page > max_pages:
            break
        data = get_one_page(page, url)
        for movie in data["results"]:
            all_data.append(get_one_movie(movie))

    return all_data


def get_one_movie(movie):
    title = movie["title"]
    release_date = movie["release_date"]
    vote_average = movie["vote_average"]
    # print(f"Title: {title}, Release Date: {release_date}, Vote Average: {vote_average}")
    return title


url = "https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&release_date.lte=01%2F01%2F2024&sort_by=popularity.desc&vote_average.gte=8"
movies = get_all_pages(total_pages, url, max_pages=100)
with open(f"{ROOT_PATH}{DATASET_PATH}movie.json", "w") as f:
    json.dump(movies, f, ensure_ascii=False, indent=4)
