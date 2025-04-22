import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials
import re

DATASET_PATH = "data/datasets/"
ROOT_PATH = "../"

PLAYLISTS_TO_QUERY = {
    "2010s": "7k8f69eOQ5Q6iL1ZU3G16Z",
    "2000s": "2qvHITZHYJHEcOMpqHcDAD",
    "90s": "3qus1xeaWG8HPdggsOoHqw",
    "80s": "0U5mlTM3kTG8JF4ueWcbRB",
    "70s": "6UQrT6GQSY0scGIAHHLAji",
}

#  API call made the 22 April 2025

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

top_songs = []
for playlist_id in PLAYLISTS_TO_QUERY.values():
    # Fetch playlist tracks
    results = sp.playlist_tracks(playlist_id, limit=100)

    # Loop through and print track info
    for i, item in enumerate(results["items"]):
        track = item["track"]
        name = track["name"]
        artist = track["artists"][0]["name"]
        song_name = f"{name} by {artist}"
        song_name = re.sub(r"\s*\(.*?\)", "", song_name)
        song_name = re.sub(r"\s*\[.*?\]", "", song_name)
        song_name = song_name.split("-")[0].strip()
        top_songs.append(song_name)


with open(f"{ROOT_PATH}{DATASET_PATH}music.json", "w") as f:
    json.dump(top_songs, f, ensure_ascii=False, indent=4)
