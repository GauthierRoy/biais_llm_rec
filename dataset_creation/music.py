import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials
import re

DATASET_PATH = "data/datasets/"
ROOT_PATH = "../"

with open(f"{ROOT_PATH}.config", "r") as f:
    lines = f.readlines()
for line in lines:
    if line.startswith("SPOTIPY_CLIENT_ID"):
        id = line.split("=")[1].strip()
    elif line.startswith("SPOTIPY_CLIENT_SECRET"):
        secret = line.split("=")[1].strip()

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(client_id=id, client_secret=secret)
)

# Search for most popular songs in the given time range
years = [
    "2004",
    "2005",
    "2006",
    "2007",
    "2008",
    "2009",
    "2010",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
]
top_songs = []

for year in years:
    results = sp.search(q=f"year:{year}", type="track", limit=50)
    for track in results["tracks"]["items"]:
        song = track["name"]
        artist = track["artists"][0]["name"]
        # remove inside parenthesis
        song_name = f"{song} - {artist}"
        song_name = re.sub(r"\s*\(.*?\)", "", song_name)
        top_songs.append(song_name)


with open(f"{ROOT_PATH}{DATASET_PATH}music.json", "w") as f:
    json.dump(top_songs, f, ensure_ascii=False, indent=4)
