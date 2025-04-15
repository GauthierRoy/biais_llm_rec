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
    "Avengers: Infinity War", "Top Gun: Maverick", "The Dark Knight", "The Matrix",
    "Inception", "Gladiator", "Spider-Man: Across the Spider-Verse",
    "The Lord of the Rings: The Return of the King", "The X-Treme Riders",
    "The Lord of the Rings: The Fellowship of the Ring", "Puss in Boots: The Last Wish",
    "Spider-Man: Into the Spider-Verse", "Avengers: Endgame",
    "The Lord of the Rings: The Two Towers", "Harry Potter and the Deathly Hallows: Part 2",
    "Princess Mononoke", "Alien", "Star Wars", "Terminator 2: Judgment Day",
    "The Good, the Bad and the Ugly", "Saving Private Ryan", "Scarface",
    "Hacksaw Ridge", "Inglourious Basterds", "Oldboy", "Django Unchained",
    "Léon: The Professional", "Zack Snyder's Justice League",
    "Demon Slayer -Kimetsu no Yaiba- The Movie: Mugen Train", "Jujutsu Kaisen 0",
    "Full Metal Jacket", "City of God", "Apocalypse Now", "Along with the Gods: The Two Worlds",
    "Ford v Ferrari", "New Gods: Nezha Reborn", "Seven Samurai",
    "Black Clover: Sword of the Wizard King", "The Empire Strikes Back",
    "Along with the Gods: The Last 49 Days", "Miraculous World: New York, United HeroeZ",
    "For a Few Dollars More", "Bound by Honor", "Mortal Kombat Legends: Scorpion's Revenge",
    "Kill Shot", "Neon Genesis Evangelion: The End of Evangelion", "Come and See",
    "I Am Nezha 2", "My Hero Academia: Heroes Rising", "Das Boot", "Elite Squad",
    "A Taxi Driver", "Evangelion: 3.0+1.0 Thrice Upon a Time",
    "Justice League Dark: Apokolips War", "The Weapon", "Ran", "Harakiri",
    "Ranma ½: The Movie 2 — The Battle of Togenkyo: Rescue the Brides!", "Green Snake",
    "Lock, Stock and Two Smoking Barrels", "The Best in Hell", "Black Butler: Book of Murder",
    "The Legend of Hei", "Sherlock Jr.", "Yojimbo", "Primal: Tales of Savagery",
    "Gridman Universe", "Digimon Adventure: Last Evolution Kizuna", "The Wages of Fear",
    "Hard Knox", "The Legend of Maula Jatt", "Initial D Legend 1: Awakening",
    "Chhota Bheem aur Krishna: Pataliputra - City of the Dead",
    "Transformers Prime: Beast Hunters - Predacons Rising", "Chasing the Storm",
    "Saga of Tanya the Evil: The Movie", "Shockwave", "XX: Beautiful Weapon",
    "Fate/strange Fake -Whispers of Dawn-", "Bungo Stray Dogs: Dead Apple", "Yellow Colt",
    "Ghosted", "Kraven Redux", "Kizumonogatari Part 3: Reiketsu",
    "Ramayana: The Legend of Prince Rama", "XXX", "Gintama: The Very Final",
    "Samurai Rebellion", "Pretty Guardian Sailor Moon Eternal the Movie Part 2",
    "Mission «Sky»", "Powderkeg", "Death by Misadventure: The Mysterious Life of Bruce Lee",
    "Memoirs of a Lady Ninja", "Kizumonogatari Part 2: Nekketsu", "Speed Dragon",
    "Girls und Panzer das Finale: Part IV", "Invincible",
    "Pretty Cure All Stars DX3: Deliver the Future! The Rainbow-Colored Flower That Connects the World",
    "The Fixer", "Against All Enemies", "The Killing Death",
    "Digimon Adventure: Our War Game", "Hot Wheels AcceleRacers: Breaking Point", "Red Dawn",
    "Hot Wheels AcceleRacers: Ignition", "Princess Principal Crown Handler: Chapter 2",
    "The Fatal Game", "Viduthalai: Part I", "Moon: The Battles of Space",
    "Warriors of the Rainbow: Seediq Bale - Part 2: The Rainbow Bridge", "The Six Devil Women",
    "The Last Bullet", "Gunpoint", "Shin Getter Robo vs Neo Getter Robo",
    "The Equalizer - The Movie: Blood & Wine", "The Odyssey", "Spider-Man!",
    "A tiro limpio", "Kamen Rider Den-O & Kiva: Climax Deka",
    "Ghost in the Shell: SAC_2045 The Last Human", "Legend of Lv Bu",
    "Pretty Cure Super Stars!", "Out of Reach", "Kamen Rider Kuuga: New Year's Dream",
    "Mask the Kekkou: Reborn", "The Kung Fu Kids VI", "Sacred Seven: Shirogane no Tsubasa",
    "Bloodsucka Jones vs. The Creeping Death", "Jackie Chan: Building an Icon", "Dadagiri",
    "Slugterra: The Emperor's Revenge", "\"Eiyuu\" Kaitai",
    "Hot Wheels AcceleRacers: The Ultimate Race", "Flash",
    "The Wandering Earth: Beyond 2020 Special Edition", "Ultraman vs. Kamen Rider",
    "Pretty Cure Miracle Leap: A Wonderful Day with Everyone", "CobraGator",
    "Slugterra: Eastern Caverns", "Kamen Rider: Run All Over the World",
    "Mega Man X: The Day of Sigma", "Yakuza of Seki", "Halo",
    "Bruce Lee: The Man and the Legend", "Smuggler's Ransom",
    "Lady Ninja Kasumi 5: Counter Attack", "Black Dog",
    "Bodacious Space Pirates: Abyss of Hyperspace", "Space Battleship Yamato",
    "Delicious Party♡Precure Movie: Dreaming♡Children's Lunch!",
    "Hot Wheels AcceleRacers: The Speed of Silence", "Bahaddur Gandu",
    "Death of Hope Part 1: Anarchy Reigns", "Kamen Rider Amazons Season 1 the Movie: Awakening",
    "Dushman Duniya Ka", "The Revenge", "Death Metal Zombies", "3-03 Rescate", "Muddat",
    "Chouriki Sentai Ohranger: The Movie", "Dragón", "Hunter × Hunter Pilot", "Torbaaz",
    "Galaxy Express 999: Eternal Traveller Emeraldas", "Deadly Game",
    "Macross Frontier: Labyrinth of Time", "Gundam Reconguista in G Movie V: Beyond the Peril of Death",
    "Princess Principal Crown Handler: Chapter 3", "Prema Yuddham",
    "Kyuukyuu Sentai GoGoFive: Sudden Shock! A New Warrior!",
    "Kikai Sentai Zenkaiger The Movie: Red Battle! All Sentai Rally!!", "M.I.A. A Greater Evil",
    "Team Hot Wheels: The Skills to Thrill", "Decisive Engagement: The Liaoxi Shenyang Campaign",
    "The Oregon Trail", "Pantai Norasingh", "Baian the Assassin, M.D.: Part 2",
    "Lord Mito: All Star Version", "Gunbuster: The Movie", "Space Runaway Ideon: Be Invoked",
    "Kamen Rider Build NEW WORLD: Kamen Rider Grease", "Deadlocked: Escape from Zone 14",
    "Seventeen Ninja 2: The Great Battle", "Full Metal Panic! Movie 2: One Night Stand",
    "Say Your Prayers... and Dig Your Grave!", "Galaxy Investigation 2100: Border Planet",
    "Escape from Tarkov. Raid.", "The Battle", "Hot Wheels: Build the Epic Race",
    "Admiral Yamamoto", "Ringgo: The Dog Shooter", "Lady Ninja Kasumi 6: Yukimura Assasination",
    "LEGO DC Super Hero Girls: Galactic Wonder", "Loha",
    "Kamen Rider THE WINTER MOVIE: Gotchard & Geats Strongest Chemy★Great Gotcha Operation",
    "Krantiveer", "Aggressive Behavior",
    "Mobile Suit Gundam Unicorn Film And Live The Final - A Mon Seul Desir",
    "Gundam Reconguista in G Movie IV: Love That Cries Out in Battle", "The Spider Returns",
    "Stephen the Great: Vaslui 1475", "Kamen Rider Ryuki Special: 13 Riders",
    "Fatal Mission", "Ninja: The Final Duel II", "Miniforce: Raid of Hamburger Monsters",
    "Fight to Survive", "The Great War", "Lady Ninja Kasumi 3: Secret Skills", "Agni I.P.S.",
    "G.I. Joe: The Revenge of Cobra", "Kikai Sentai Zenkaiger vs. Kiramager vs. Senpaiger",
    "Sherdil", "Slugterra: Into The Shadows", "Praetorian", "Ganryujima: Kojiro and Musashi",
    "Aayirathil Iruvar", "Sand Land", "Tsubasa Chronicle: Shunraiki", "The Escape Plan",
    "Tokyo Revengers: Bloody Halloween", "The Powerpuff Girls Rule!!!",
    "Girls in Trouble: Space Squad Episode Zero", "The Final Game of Death", "Angrakshak",
    "Thunder Chase", "Oda Nobunaga", "ThunderCats: Exodus", "Last Stand at Lang Mei",
    "Kamen Rider Amazons Season 2 the Movie: Reincarnation"
])

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