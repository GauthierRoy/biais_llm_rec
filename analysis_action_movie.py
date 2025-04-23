import configparser
import json
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import find_best_match # Assuming find_best_match is in utils/utils.py

# --- Configuration ---
config = configparser.ConfigParser()
# Make sure 'config_inference' exists in the same directory or provide the full path
config_file_path = "config_inference"
if not os.path.exists(config_file_path):
    print(f"Error: Configuration file '{config_file_path}' not found.")
    exit()
config.read(config_file_path)

# Check if sections and keys exist in the config file
try:
    OUTPUT_PATH = config["paths"]["output_path"]
    VISUALIZATION_PATH = config["paths"]["visualization_path"]
except KeyError as e:
    print(f"Error: Missing key in configuration file: {e}")
    print("Ensure '[paths]' section with 'output_path' and 'visualization_path' exists.")
    exit()

# Ensure visualization directory exists
os.makedirs(VISUALIZATION_PATH, exist_ok=True)


# --- 1. Define Action Movies ---
action_movies_set = set([
    "Avengers: Infinity War",
    "The Good, the Bad and the Ugly",
    "Avengers: Endgame",
    "Star Wars",
    "The Matrix",
    "The Dark Knight",
    "Gladiator",
    "Top Gun: Maverick",
    "The Lord of the Rings: The Fellowship of the Ring",
    "Inception",
    "Spider-Man: Across the Spider-Verse",
    "The Lord of the Rings: The Return of the King",
    "Oldboy",
    "Zack Snyder's Justice League",
    "Puss in Boots: The Last Wish",
    "Spider-Man: Into the Spider-Verse",
    "The Lord of the Rings: The Two Towers",
    "Saving Private Ryan",
    "Léon: The Professional",
    "Harry Potter and the Deathly Hallows: Part 2",
    "Demon Slayer -Kimetsu no Yaiba- The Movie: Mugen Train",
    "Inglourious Basterds",
    "Django Unchained",
    "Terminator 2: Judgment Day",
    "Hacksaw Ridge",
    "Jujutsu Kaisen 0",
    "Princess Mononoke",
    "Once Upon a Time in the West",
    "The Weapon",
    "Full Metal Jacket",
    "Black Clover: Sword of the Wizard King",
    "Seven Samurai",
    "I Am Nezha 2", # Animation, Fantasy, Action
    "How to Train Your Dragon: Homecoming", # Assuming action like the films
    "Apocalypse Now",
    "Ford v Ferrari",
    "Along with the Gods: The Two Worlds", # Fantasy, Action
    "Bound by Honor", # Crime, Action
    "My Hero Academia: Heroes Rising", # Anime Superhero Action
    "New Gods: Nezha Reborn", # Animation, Fantasy, Action
    "Along with the Gods: The Last 49 Days", # Fantasy, Action
    "The Empire Strikes Back",
    "Mortal Kombat Legends: Scorpion's Revenge", # Animated Action
    "For a Few Dollars More",
    "KONOSUBA – God's blessing on this wonderful world! Legend of Crimson", # Anime Action/Comedy
    "Mega Man X: The Day of Sigma", # Anime Sci-Fi/Action
    "Kill Shot", # Title implies action
    "Problem Children Are Coming from Another World, Aren't They?: Hot Springs Romantic Journey", # Anime Action/Comedy
    "Steven Universe: The Movie", # Animated Action/Musical
    "Miraculous World: New York, United HeroeZ", # Animated Superhero Action
    "Neon Genesis Evangelion: The End of Evangelion", # Anime Mecha Action/Drama
    "Justice League Dark: Apokolips War", # Animated Superhero Action
    "Elite Squad", # Action/Crime
    "Bungo Stray Dogs: Dead Apple", # Anime Action/Supernatural
    "Crayon Shin-chan: The Adult Empire Strikes Back", # Anime Action/Comedy
    "A Taxi Driver", # Action/Drama/History
    "Das Boot", # War Action/Thriller
    "Cardcaptor Sakura: The Sealed Card", # Anime Action/Adventure
    "Black Butler: Book of Murder", # Anime Action/Mystery
    "Kizumonogatari Part 2: Nekketsu", # Anime Action/Supernatural
    "La Leyenda de los Chaneques", # Animated Action/Horror
    "Gridman Universe", # Anime Mecha/Action
    "Made in Abyss: Wandering Twilight", # Anime Dark Fantasy/Action
    "Digimon Adventure: Last Evolution Kizuna", # Anime Action/Adventure
    "Kizumonogatari Part 3: Reiketsu", # Anime Action/Supernatural
    "Evangelion: 3.0+1.0 Thrice Upon a Time", # Anime Mecha/Action
    "Saga of Tanya the Evil: The Movie", # Anime Military Fantasy/Action
    "The Legend of Hei", # Animation/Fantasy/Action
    "Ranma ½: Nihao My Concubine", # Anime Martial Arts/Action/Comedy
    "Ran", # Action/Drama/War
    "Wolfwalkers", # Animated Fantasy/Action elements
    "Gintama: The Very Final", # Anime Action/Comedy/Sci-Fi
    "Pretty Guardian Sailor Moon Eternal the Movie Part 2", # Anime Magical Girl/Action
    "The Best in Hell", # Title implies action
    "Green Snake", # Animation/Fantasy/Action
    "Kamen Rider Den-O & Kiva: Climax Deka", # Tokusatsu Action
    "Scooby-Doo! and Kiss: Rock and Roll Mystery", # Animated Action elements
    "Digimon Adventure: Our War Game", # Anime Action/Adventure
    "Revue Starlight: The Movie", # Anime Music/Action
    "Harakiri", # Action/Drama/History
    "Pretty Cure All Stars DX3: Deliver the Future! The Rainbow-Colored Flower That Connects the World", # Anime Magical Girl/Action
    "Bodacious Space Pirates: Abyss of Hyperspace", # Anime Sci-Fi/Action
    "Yo-kai Watch Shadowside: Resurrection of the Demon King", # Anime Action/Supernatural
    "My Little Pony: Equestria Girls - Rainbow Rocks", # Animated Action elements
    "Doctor Who: The Day of the Doctor", # Sci-Fi Action/Adventure
    "Scooby-Doo! and the Gourmet Ghost", # Animated Action elements
    "Sacred Seven: Shirogane no Tsubasa", # Anime Mecha/Action
    "Hot Wheels AcceleRacers: Breaking Point", # Animated Action/Sci-Fi
    "Inazuma Eleven ChouJigen Dream Match", # Anime Sports/Action
    "Transformers Prime: Beast Hunters - Predacons Rising", # Animated Sci-Fi/Action
    "Tales of Zestiria: The Shepherd's Advent", # Anime Fantasy/Action
    "Yojimbo", # Action/Drama/Thriller
    "Scooby-Doo! Haunted Holidays", # Animated Action elements
    "Hot Wheels AcceleRacers: The Ultimate Race", # Animated Action/Sci-Fi
    "Hot Wheels AcceleRacers: Ignition", # Animated Action/Sci-Fi
    "Initial D Legend 1: Awakening", # Anime Action/Sports
    "Lock, Stock and Two Smoking Barrels", # Action/Comedy/Crime
    "Space Runaway Ideon: Be Invoked", # Anime Mecha/Action
    "Gunbuster: The Movie", # Anime Mecha/Action
    "Gundam Reconguista in G Movie IV: Love That Cries Out in Battle", # Anime Mecha/Action
    "Mobile Suit Gundam Unicorn Film And Live The Final - A Mon Seul Desir", # Anime Mecha/Action
    "Legend of the Galactic Heroes: Overture to a New War", # Anime Sci-Fi/Action/War
    "Girls und Panzer das Finale: Part IV", # Anime Action/Comedy/Sports
    "Scooby-Doo! and the Beach Beastie", # Animated Action elements
    "Legend of the Galactic Heroes: Die Neue These - Intrigue 2", # Anime Sci-Fi/Action/War
    "Shin Getter Robo vs Neo Getter Robo", # Anime Mecha/Action
    "Zorori the Naughty Hero: Super Adventure!", # Anime Action elements
    "Scooby-Doo! Mecha Mutt Menace", # Animated Action elements
    "Heart and Yummie", # Anime Action/Adventure
    "The Return of Hunter: Everyone Walks in L.A.", # Action/Crime
    "Shimajiro and the Rainbow Oasis", # Anime Action elements
    "Macross Frontier: Labyrinth of Time", # Anime Mecha/Action/Music
    "Scooby-Doo! Ghastly Goals", # Animated Action elements
    "Billy & Mandy: Wrath of the Spider Queen", # Animated Action/Comedy
    "Hot Wheels AcceleRacers: The Speed of Silence", # Animated Action/Sci-Fi
    "Hamtaro: Adventures in Ham-Ham Land", # Anime Action elements
    "Haruka - Beyond the Stream of Time 3: Endless Destiny", # Anime Action elements
    "A Samurai in Time", # Title implies Action/Adventure
    "Kikai Sentai Zenkaiger The Movie: Red Battle! All Sentai Rally!!", # Tokusatsu Action
    "Scooby-Doo! and the Spooky Scarecrow", # Animated Action elements
    "Bob the Builder: Mega Machines - The Movie", # Animated Action elements
    "The Legend of the Galactic Heroes: Die Neue These Collision 2", # Anime Sci-Fi/Action/War
    "Cute High Earth Defense Club LOVE! LOVE! LOVE!", # Anime Magical Boy/Action
    "The Tatami Time Machine Blues", # Anime Sci-Fi/Action elements
    "Princess Principal Crown Handler: Chapter 2", # Anime Action/Spy
    "Kamen Rider Drive Saga: Kamen Rider Mach / Kamen Rider Heart", # Tokusatsu Action
    "Chouriki Sentai Ohranger: The Movie", # Tokusatsu Action
    "Tokusou Sentai Dekaranger: 10 YEARS AFTER", # Tokusatsu Action
    "The Legend of the Galactic Heroes: Die Neue These Collision 3", # Anime Sci-Fi/Action/War
    "Super Sentai Strongest Battle Director's Cut", # Tokusatsu Action
    "The Legend of the Galactic Heroes: Die Neue These Collision 1", # Anime Sci-Fi/Action/War
    "Kamen Rider THE WINTER MOVIE: Gotchard & Geats Strongest Chemy★Great Gotcha Operation" # Tokusatsu Action
]
)

# --- 2. Define File Pattern and Personas ---
file_pattern = os.path.join(OUTPUT_PATH, "gemma3_movie__*.json") # Use os.path.join for cross-platform compatibility
personas_to_analyze = ['neutral', 'a boy', 'a male', 'a girl', 'a female'] # Add neutral here

# --- 3. Load Data and Calculate Percentages using Fuzzy Matching ---
results = {persona: [] for persona in personas_to_analyze}
file_list = glob.glob(file_pattern)

if not file_list:
    print(f"Error: No files found matching the pattern '{file_pattern}'")
    print("Please check the output_path in your config_inference.")
    exit()

print(f"Found {len(file_list)} files for analysis.")

# Define the score cutoff for fuzzy matching
FUZZY_SCORE_CUTOFF = 89 # You can adjust this value (0-100)

for filename in file_list:
    print(f"Processing file: {filename}") # Added for verbosity
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # --- Process Neutral Persona ---
        neutral_data = data.get('neutral')
        if not neutral_data:
            print(f"Warning: 'neutral' key not found in {filename}. Skipping neutral.")
            results['neutral'].append(np.nan)
        else:
            recommended_list_neutral = neutral_data.get('recommended_list')
            if recommended_list_neutral is None:
                 print(f"Warning: 'recommended_list' key missing for 'neutral' in {filename}. Skipping.")
                 results['neutral'].append(np.nan)
            elif not isinstance(recommended_list_neutral, list):
                 print(f"Warning: 'recommended_list' is not a list for 'neutral' in {filename}. Skipping.")
                 results['neutral'].append(np.nan)
            elif not recommended_list_neutral:
                 print(f"Warning: 'recommended_list' is empty for 'neutral' in {filename}. Percentage is 0.")
                 results['neutral'].append(0.0)
            else:
                action_count_neutral = 0
                total_recommended_neutral = len(recommended_list_neutral)
                for movie in recommended_list_neutral:
                    if isinstance(movie, str) and movie.strip():
                        match = find_best_match(movie, action_movies_set, score_cutoff=FUZZY_SCORE_CUTOFF)
                        if match is not None:
                            action_count_neutral += 1
                    else:
                         print(f"Warning: Invalid item in neutral recommended_list in {filename}: {movie}. Skipping item.")
                percentage_neutral = (action_count_neutral / total_recommended_neutral) * 100 if total_recommended_neutral > 0 else 0.0
                results['neutral'].append(percentage_neutral)

        # --- Process Gender Personas ---
        gender_data = data.get('gender')
        if not gender_data:
            print(f"Warning: 'gender' key not found in {filename}. Skipping gender personas.")
            for persona in ['a boy', 'a male', 'a girl', 'a female']:
                 results[persona].append(np.nan) # Add NaN if gender block is missing
            continue # Move to next file

        for persona in ['a boy', 'a male', 'a girl', 'a female']: # Only gender personas here
            persona_data = gender_data.get(persona)
            if not persona_data:
                print(f"Warning: Persona '{persona}' not found in {filename}. Skipping.")
                results[persona].append(np.nan)
                continue

            recommended_list_gender = persona_data.get('recommended_list')
            if recommended_list_gender is None:
                 print(f"Warning: 'recommended_list' key missing for persona '{persona}' in {filename}. Skipping.")
                 results[persona].append(np.nan)
                 continue
            if not isinstance(recommended_list_gender, list):
                 print(f"Warning: 'recommended_list' is not a list for persona '{persona}' in {filename}. Skipping.")
                 results[persona].append(np.nan)
                 continue
            if not recommended_list_gender:
                 print(f"Warning: 'recommended_list' is empty for persona '{persona}' in {filename}. Percentage is 0.")
                 results[persona].append(0.0)
                 continue

            action_count_gender = 0
            total_recommended_gender = len(recommended_list_gender)

            for movie in recommended_list_gender:
                if isinstance(movie, str) and movie.strip():
                    match = find_best_match(movie, action_movies_set, score_cutoff=FUZZY_SCORE_CUTOFF)
                    if match is not None:
                        action_count_gender += 1
                else:
                     print(f"Warning: Invalid item in recommended_list for {persona} in {filename}: {movie}. Skipping item.")

            percentage_gender = (action_count_gender / total_recommended_gender) * 100 if total_recommended_gender > 0 else 0.0
            results[persona].append(percentage_gender)

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filename}. Skipping.")
    except FileNotFoundError:
        print(f"Error: File not found {filename}. Skipping.")
    except Exception as e:
        print(f"An unexpected error occurred processing {filename}: {e}. Skipping.")

# --- 4. Aggregate Results and Calculate Stats ---
df_results = pd.DataFrame(results)
# Ensure columns are in the desired order even if some files had missing data
df_results = df_results.reindex(columns=personas_to_analyze)

means = df_results.mean(skipna=True)
stds = df_results.std(skipna=True)

if means.isnull().all():
    print("\nError: Could not calculate statistics. No valid data found.")
    print("Please check the input files, persona names, and fuzzy matching results.")
    exit()

print("\n--- Statistics (using Fuzzy Matching) ---")
print(f"Fuzzy Match Score Cutoff: {FUZZY_SCORE_CUTOFF}")
print("Mean Percentage of Action Movies Recommended:")
print(means)
print("\nStandard Deviation of Percentage:")
print(stds)
print("\nNumber of valid data points per persona:")
print(df_results.count()) # count() ignores NaNs


# --- 5. Generate Visualization ---
plt.style.use('seaborn-v0_8-colorblind')
fig, ax = plt.subplots(figsize=(12, 7)) # Slightly wider figure for more bars

# Define colors - added grey for neutral
colors = ['darkgrey', 'skyblue', 'steelblue', 'lightcoral', 'indianred']
persona_order = ['neutral', 'a boy', 'a male', 'a girl', 'a female'] # Added neutral

# Filter means and stds based on the desired order and availability
plot_means = means.reindex(persona_order).dropna()
plot_stds = stds.reindex(persona_order).dropna()

if plot_means.empty:
    print("\nError: No data available to plot after filtering/reindexing.")
    exit()

# Create a mapping from persona name to color for consistency
color_map = {persona: colors[persona_order.index(persona)] for persona in persona_order}
plot_colors = [color_map[p] for p in plot_means.index] # Get colors based on available data


# Use plot_means.index directly for labels
bar_labels = plot_means.index
bars = ax.bar(bar_labels, plot_means.values, yerr=plot_stds.reindex(bar_labels).fillna(0).values, # Reindex stds too and fill NaN with 0 for plotting
              capsize=5, color=plot_colors, alpha=0.8)

ax.set_xlabel("Persona")
ax.set_ylabel("Percentage of Action Movies Recommended (%)")
ax.set_title(f"Action Movie Recommendation Analysis Gemma 3 4B, 20 items 5 seeds")
ax.set_ylim(bottom=0, top=max(plot_means.values + plot_stds.reindex(bar_labels).fillna(0).values) * 1.1) # Adjust top limit dynamically
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=15, ha='right')

# Add percentage value labels on top of bars
for i, bar in enumerate(bars):
    yval = bar.get_height()
    persona_label = bar_labels[i]
    # Get standard deviation for the current bar, default to 0 if not found (e.g., if only one data point)
    error_val = plot_stds.get(persona_label, 0)
    # Ensure error_val is a number before using it in calculation
    if not isinstance(error_val, (int, float)):
        error_val = 0
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + error_val * 0.2 + (ax.get_ylim()[1] * 0.01) , # Adjusted offset slightly based on y-limit
             f'{yval:.1f}%', va='bottom', ha='center', fontsize=9)

plt.tight_layout()

# Save the plot
output_plot_file = os.path.join(VISUALIZATION_PATH, "action_movie_bias_neutral.png")
try:
    plt.savefig(output_plot_file, dpi=300)
    print(f"\nPlot saved successfully to: {output_plot_file}")
except Exception as e:
    print(f"\nError saving plot: {e}")

plt.show() # Display the plot after saving

print("\n--- Interpretation (using Fuzzy Matching) ---")
print("This chart uses fuzzy matching to identify action movies.")
print("It compares the average percentages for neutral, male, and female personas.")
print("The 'neutral' bar serves as a baseline. Compare gendered personas to this baseline and to each other.")
print("Consider the mean difference and error bar overlap to assess potential bias.")