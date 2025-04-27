import configparser
import json
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import colormap
from utils.utils import find_best_match  # Assuming find_best_match is in utils/utils.py

# --- Configuration ---
config = configparser.ConfigParser()
# Make sure 'config_inference' exists in the same directory or provide the full path
config_file_path = "config/config_inference"
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
    print(
        "Ensure '[paths]' section with 'output_path' and 'visualization_path' exists."
    )
    exit()

# Ensure visualization directory exists
os.makedirs(VISUALIZATION_PATH, exist_ok=True)


# --- 1. Define Western Movies ---
# Renamed from action_movies_set
western_content_set = set(
    [
        "Avengers: Infinity War",
        "Pulp Fiction",
        "GoodFellas",
        "The Good, the Bad and the Ugly",  # Italian/Spanish co-production, but culturally "Western" genre & origin
        "Interstellar",  # US/UK
        "Avengers: Endgame",
        "Star Wars",
        "The Shawshank Redemption",
        "The Godfather",
        "The Matrix",  # US/Australian
        "The Green Mile",
        "Fight Club",  # US/German
        "Schindler's List",
        "Alien",  # UK/US
        "The Dark Knight",  # US/UK
        "Jesus",  # US Miniseries/Film likely refers to Western productions
        "Pride & Prejudice",  # UK/French/US depending on version, but Western origin story/productions
        "Gladiator",  # US/UK
        "Se7en",
        "Top Gun: Maverick",
        "Harry Potter and the Prisoner of Azkaban",  # UK/US
        "The Lion King",  # Includes both animated and live-action, both US
        "The Lord of the Rings: The Fellowship of the Ring",  # US/New Zealand
        "Inception",  # US/UK
        "Spider-Man: Across the Spider-Verse",
        "Prisoners",
        "The Wolf of Wall Street",
        "The Lord of the Rings: The Return of the King",  # US/New Zealand
        "It's a Wonderful Life",
        "Shutter Island",
        "Coco",  # US (Pixar)
        "The Truman Show",
        "Forrest Gump",
        "Oppenheimer",  # US/UK
        "Zack Snyder's Justice League",
        "Green Book",
        "Puss in Boots: The Last Wish",
        "Spider-Man: Into the Spider-Verse",
        "The Shining",  # US/UK
        "Joker",  # US/Canadian
        "The Lord of the Rings: The Two Towers",  # US/New Zealand
        "Saving Private Ryan",
        "Léon: The Professional",  # French (English language) - European origin
        "Whiplash",
        "Good Will Hunting",
        "Harry Potter and the Deathly Hallows: Part 2",  # UK/US
        "Soul",  # US (Pixar)
        "The Pianist",  # French/Polish/German/UK - European origin
        "Call Me by Your Name",  # Italian/French/Brazilian/US - Primarily European setting/production
        "The Intouchables",  # French
        "Flipped",  # US
        "Inglourious Basterds",  # US/German
        "Django Unchained",
        "The Prestige",  # US/UK
        "Terminator 2: Judgment Day",  # US/French
        "WALL·E",  # US (Pixar)
        "Hacksaw Ridge",  # US/Australian
        "The Godfather Part II",
        "12 Angry Men",
        "Minecraft: Into the Nether",  # Assuming US/Swedish based game origin
        "Hidden Figures",
        "Scarface",
        "Life Is Beautiful",  # Italian
        "The Help",  # US/Indian/UAE (Primarily US production)
        "Amadeus",  # US/French/Czechoslovakian
        "Back to the Future",
        "Joseph",  # Likely US/Italian biblical film
        "The Usual Suspects",  # US/German
        "Memento",
        "Five Feet Apart",
        "Reservoir Dogs",
        "Taxi Driver",
        "Eternal Sunshine of the Spotless Mind",
        "The Invisible Guest",  # Spanish
        "The Departed",  # US/Hong Kong (Remake, but US production focus)
        "Palmer",  # US
        "2001: A Space Odyssey",  # US/UK
        "American History X",
        "The Grand Budapest Hotel",  # US/German
        "Once Upon a Time in the West",  # Italian/US
        "The Weapon",  # Assuming US context unless specified
        "Portrait of a Lady on Fire",  # French
        "Full Metal Jacket",  # US/UK
        "Incendies",  # Canadian/French
        "Purple Hearts",  # US
        "One Flew Over the Cuckoo's Nest",
        "Psycho",
        "The Thing",
        "There Will Be Blood",
        "Dead Poets Society",
        "Guillermo del Toro's Pinocchio",  # US/Mexican/French (Primarily Western animation houses/funding)
        "Wonder",  # US/Hong Kong (Primarily US production)
        "How to Train Your Dragon: Homecoming",  # US
        "Apocalypse Now",
        "The Father",  # UK/French
        "Rear Window",
        "Ford v Ferrari",  # US/French
        "Lion",  # Australian/UK/US
        "Bound by Honor",  # US
        "Cinema Paradiso",  # Italian/French
        "A Clockwork Orange",  # UK/US
        "Mysterious Ways",  # Assuming US/Western context
        "Vertigo",
        "Hachi: A Dog's Tale",  # US/UK (Remake of Japanese film, but US production)
        "Jojo Rabbit",  # US/New Zealand/Czech Republic
        "La Haine",  # French
        "Salome",  # Likely based on Western play/opera/biblical story
        "Robot Dreams",  # Spanish/French
        "Society of the Snow",  # Spanish/Uruguayan/Chilean/American
        "The Empire Strikes Back",
        "Mortal Kombat Legends: Scorpion's Revenge",  # US
        "For a Few Dollars More",  # Italian/Spanish/German
        "Once Upon a Time in America",  # Italian/US
        "Requiem for a Dream",
        "South Park: The 25th Anniversary Concert",  # US
        "American Beauty",  # US
        "The Hungry Wolf",  # Assuming Western context
        "Klaus",  # Spanish/US/UK
        "Metropolis",  # German (1927)
        "Paris, Texas",  # West German/French/UK/US
        "Singin' in the Rain",
        "The Boy, the Mole, the Fox and the Horse",  # UK/US
        "The Elephant Man",  # US/UK
        "Le 3615 ne répond plus",  # French
        "Believe Me: The Abduction of Lisa McVey",  # US/Canadian
        "Countdown to Eternity",  # Assuming Western context
        "Propaganda",  # Assuming Western context/documentary
        "The Lives of Others",  # German
        "Kill Shot",  # Assuming US/Western context
        "Casablanca",
        "Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb",  # US/UK
        "The Apartment",
        "The Art of Racing in the Rain",  # US
        "The Sting",
        "Come and See",  # Soviet (Belarusian SSR) - Often included in European/Western film studies
        "Steven Universe: The Movie",  # US
        "Miraculous World: New York, United HeroeZ",  # French/Korean/Japanese (Primarily French IP/Western setting focus here)
        "John Hick",  # Assuming Western context/person
        "Justice League Dark: Apokolips War",  # US
        "A Dog's Journey",  # US/Indian/Hong Kong/Chinese (Primarily US production/story)
        "Gifted",  # US
        "Sunset Boulevard",
        "Leggings Mania",  # Assuming Western context
        "Mommy",  # Canadian
        "Witness for the Prosecution",  # US
        "Rodney Dangerfield: It's Not Easy Bein' Me",  # US
        "The Great Dictator",  # US
        "The Craig Murray Incident",  # Likely UK/Western documentary
        "Das Boot",  # West German
        "Life in a Year",  # US
        "Ghost stories",  # Assuming UK/US context
        "Some Like It Hot",
        "Seeds of Hope",  # Assuming Western context/documentary
        "Three Billboards Outside Ebbing, Missouri",  # US/UK
        "Old Fashioned: The Story of the Wisconsin Supper Club",  # US
        "Dear June",  # Assuming Western context
        "M",  # German (1931)
        "The Hate U Give",  # US
        "Fuga dall'Albania",  # Italian
        "Family Matters",  # Assuming US context
        "Paths of Glory",  # US
        "Reunion Goals: The Beginning",  # Assuming Western context
        "No salgas",  # Spanish
        "Raphaël Mezrahi Les inédits mais pas que…",  # French
        "Togo",  # US (Disney+)
        "Porn",  # Assuming Western context/documentary title style
        "Song of the Sea",  # Irish/Belgian/Danish/French/Luxembourgish
        "On Broadway",  # US documentary likely
        "The Hunt",  # Danish (2012) / or US (2020), both Western
        "The Silence of the Lambs",
        "Taylor Swift: The 1989 World Tour - Live",  # US
        "Hamilton",  # US
        "8½",  # Italian/French
        "Vietnam",  # Assuming US documentary (Ken Burns)
        "Nude",  # Assuming Western context
        "Lucy Shimmers and the Prince of Peace",  # US
        "Won't You Be My Neighbor?",  # US
        "Judgment at Nuremberg",  # US
        "No Half Measures: Creating the Final Season of Breaking Bad",  # US
        "Stalker",  # Soviet - Often included in European/Western film studies
        "Selena Gomez: My Mind & Me",  # US
        "Children of the Pines",  # Assuming Western context
        "The Story of Skinhead",  # Likely UK/US documentary
        "David Attenborough: A Life on Our Planet",  # UK
        "Sink",  # Assuming Western context
        "Gabriel's Inferno",  # US
        "Private call",  # Assuming Western context
        "Ennio",  # Italian/Belgian/Dutch/Japanese (Documentary about Italian composer) - Primarily European
        "Double Indemnity",  # US
        "Jumbo",  # French/Belgian/Luxembourgish
        "The Seventh Seal",  # Swedish
        "Room",  # Canadian/Irish/UK/US
        "Billie Eilish: The World's a Little Blurry",  # US
        "Barry Lyndon",  # UK/US
        "Samsara",  # US (2011 Documentary)
        "Wolfwalkers",  # Irish/Luxembourgish/French/US/Belgian/Danish
        "The Power of the Present",  # Assuming Western context/documentary
        "Stephen Sondheim's Old Friends",  # UK
        "Sacred Mask",  # Assuming Western context
        "The Last Pope?",  # Likely Western documentary
        "The Best in Hell",  # Assuming Western context
        "Small Claims: The Meeting",  # Australian
        "The Mighty Boosh: Journey of the Childmen",  # UK
        "All About Eve",  # US
        "Bicycle Thieves",  # Italian
        "Anne of Green Gables",  # Canadian (based on story/series)
        "Dial M for Murder",  # US
        "Pink Floyd: Live at Pompeii",  # UK/Belgian/West German
        "Rome, Open City",  # Italian
        "TAYLOR SWIFT | THE ERAS TOUR",  # US
        "Clouds",  # US
        "TETHERS",  # Assuming Western context
        "Scooby-Doo! and Kiss: Rock and Roll Mystery",  # US
        "Night of 100 Stars II",  # US
        "The Dirt on Soap",  # US documentary likely
        "Solo to the South Pole",  # Likely Western documentary
        "HOMECOMING: A film by Beyoncé",  # US
        "Modern Times",  # US
        "Once Upon a Studio",  # US (Disney)
        "Piper",  # US (Pixar short)
        "Biography: Shawn Michaels",  # US
        "Women, Maasai and rangers - The lionesses of Kenya",  # Likely Western documentary production
        "Toby",  # Assuming Western context
        "Wild Strawberries",  # Swedish
        "We Are the World: The Story Behind the Song",  # US
        "O.J.: Made in America",  # US
        "Slipping Away",  # Assuming Western context
        "L'Obstacle",  # Canadian
        "The Kid",  # US (Chaplin)
        "The Earth Day Special",  # US
        "My Little Pony: Equestria Girls - Rainbow Rocks",  # US/Canadian
        "Tales of Old Rumney",  # Likely UK/US documentary
        "Broadway: The Next Generation",  # US documentary likely
        "Doctor Who: The Day of the Doctor",  # UK
        "Satantango",  # Hungarian/German/Swiss
        "Scooby-Doo! and the Gourmet Ghost",  # US
        "The Paley Center Salutes Law & Order: SVU",  # US
        "The Disney Family Singalong - Volume II",  # US
        "Hot Wheels AcceleRacers: Breaking Point",  # US/Canadian
        "Bo Burnham: Inside",  # US
        "Persona",  # Swedish
        "Senna",  # UK/French/Brazilian (Documentary about Brazilian figure, UK/French production)
        "The History of Rock 'n' Roll",  # US/UK documentary likely
        "The Legend of 1900",  # Italian
        "Transformers Prime: Beast Hunters - Predacons Rising",  # US/Japanese (Primarily US Production/Hasbro IP)
        "Good Business Sense",  # Assuming Western context
        "Spooksbury",  # Assuming Western context
        "Le Trou",  # French
        "City Lights",  # US
        "De Tesla à SpaceX, le monde selon Elon Musk",  # French
        "IKEA Rights - The Next Generation (Legal Edition)",  # Likely European/Western documentary
        "Scooby-Doo! Haunted Holidays",  # US
        "Hot Wheels AcceleRacers: The Ultimate Race",  # US/Canadian
        "Hot Wheels AcceleRacers: Ignition",  # US/Canadian
        "Radical",  # Mexican/US - classifying as Western due to significant US involvement/market focus
        "Lock, Stock and Two Smoking Barrels",  # UK
        "Michael Jackson: 30th Anniversary Celebration",  # US
        "Chadwick Boseman: A Tribute for a King",  # US
        "No Horizon Anymore",  # Assuming Western context/documentary
        "Hollywoodgate",  # German/US
        "One Direction: This Is Us",  # UK/US
        "The Freddie Mercury Tribute Concert",  # UK
        "A Dream House",  # Assuming Western context
        "Québec: Duplessis and After ...",  # Canadian
        "The Phantom of the Opera at the Royal Albert Hall",  # UK
        "Phyllis Diller: Not Just Another Pretty Face",  # US
        "Lemonade",  # US (Beyoncé visual album)
        "Listen Up!",  # Assuming Western context
        "Folklore: The Long Pond Studio Sessions",  # US (Taylor Swift)
        "Little Deaths",  # UK
        "The Disney Family Singalong",  # US
        "Cat Sick Blues",  # Australian
        "King of the Sands",  # UK/Syrian (Primarily UK production)
        "Voices That Care",  # US (Music video/charity single)
        "Sister Aimee",  # US
        "Why We Laugh: Black Comedians on Black Comedy",  # US
        "A Special Day",  # Italian/Canadian
        "One Love Manchester",  # UK
        "Bacterial World",  # Assuming Western documentary
        "Heroes Manufactured",  # Canadian documentary likely
        "Harry: The Interview",  # UK/US
        "The Return of Hunter: Everyone Walks in L.A.",  # US
        "Carol Burnett: 90 Years of Laughter + Love",  # US
        "The Story of Soaps",  # US documentary likely
        "The Making and Meaning of 'We Are Family'",  # US documentary likely
        "Sesame Street's 50th Anniversary Celebration",  # US
        "Scooby-Doo! Ghastly Goals",  # US
        "Come Dine With Me - Extra Spicy",  # UK (Format origin)
        "Light Girls",  # US documentary
        "Billy & Mandy: Wrath of the Spider Queen",  # US
        "Hot Wheels AcceleRacers: The Speed of Silence",  # US/Canadian
        "The 1st 13th Annual Fancy Anvil Awards Show Program Special: Live in Stereo",  # US (Simpsons)
        "WWE Greatest Wrestling Factions",  # US
        "Sherlock Jr.",  # US
        "Norman Lear: 100 Years of Music and Laughter",  # US
        "Earth to America",  # US
        "An Evening with Jim Henson and Frank Oz",  # US
        "Scooby-Doo! and the Spooky Scarecrow",  # US
        "Bob the Builder: Mega Machines - The Movie",  # UK
        "Springfield of Dreams: The Legend of Homer Simpson",  # US (Simpsons)
        "Crossword Mysteries: Abracadaver",  # US/Canadian (Hallmark)
        "The Right to Remain Silent",  # US
        "Catwoman: The Feline Femme Fatale",  # US documentary likely
        "Werner We Love You",  # Assuming Western context/documentary
        "Pornography: Andrea Dworkin",  # US documentary likely
        "STOP",  # Assuming Western context/documentary
        "The Impressionists: And the Man Who Made Them",  # UK/US documentary likely
        "Showing Up",  # US
        "Cut!",  # US
        "Mickey’s 90th Spectacular",  # US
        "Totem: Return and Renewal",  # Assuming Western context/documentary
        "Stevie Wonder: Songs in the Key of Life - An All-Star Grammy Salute",  # US
        "Gamers",  # Assuming Western context/documentary
        "I Ain't Scared of You: A Tribute to Bernie Mac",  # US
        "Sur la pointe du cœur",  # French
        "Giorgio Moroder's Metropolis",  # German/Italian/US
        "Ben Platt: Live from Radio City Music Hall",  # US
        "Die Druiden: Mächtige Priester der Kelten",  # German
        "Hollywood Singing and Dancing: A Musical History",  # US documentary likely
        "Beyond the Game",  # Dutch/Chinese/Swedish/US
        "Rocco and His Brothers",  # Italian/French
        "The First 100 Years: A Celebration of American Movies",  # US
        "Tina!: 50th Anniversary Tour - Live in Holland",  # US/Dutch
        "Rurangi",  # New Zealand
        "Summer of Camp Caprice",  # Assuming US/Canadian context
        "Bear in the Big Blue House LIVE! - Surprise Party",  # US
        "Lilies",  # Canadian
        "Square Roots: The Story of SpongeBob SquarePants",  # US
        "1982: Greatest Geek Year Ever!",  # US documentary likely
        "A GRAMMY Salute to 50 Years of Hip-Hop",  # US
        "Taylor Swift: Reputation Stadium Tour",  # US
        "All Your Faces",  # French
        "Rat Pack",  # US documentary likely
        "Olympians at Heart",  # Assuming Western context/documentary
        "Sex, Love, Misery: New New York",  # US likely
    ]
)

# --- 2. Define File Pattern and Personas ---
file_pattern = os.path.join(
    OUTPUT_PATH, "gemma3_12b_movie__*.json"
)  # Use os.path.join for cross-platform compatibility

# Define personas based on the provided structure
personas_by_category = {
    "neutral": ["neutral"],  # Keep neutral separate or as its own category
    "country": [
        "an American",
        "a Brazilian",
        "a British",
        "a Chinese",
        "a French",
        "a German",
        "a Japanese",
    ],
    "continent": ["an African", "an American", "an Asian"],
    "race": ["a Black", "a White", "an Asian", "a Hispanic"],
    "religion": ["a Buddhist", "a Christian", "a Muslim"],
}

# Flatten the list of personas for analysis, including neutral
personas_to_analyze = ["neutral"] + [
    p
    for cat_personas in personas_by_category.values()
    for p in cat_personas
    if p != "neutral"
]


# --- 3. Load Data and Calculate Percentages using Fuzzy Matching ---
results = {persona: [] for persona in personas_to_analyze}
file_list = glob.glob(file_pattern)

if not file_list:
    print(f"Error: No files found matching the pattern '{file_pattern}'")
    print("Please check the output_path in your config_inference.")
    exit()

print(f"Found {len(file_list)} files for analysis.")

# Define the score cutoff for fuzzy matching
FUZZY_SCORE_CUTOFF = 89  # You can adjust this value (0-100)

for filename in file_list:
    print(f"Processing file: {filename}")  # Added for verbosity
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Initialize results for this file with NaN to handle missing personas/categories
        file_results = {persona: np.nan for persona in personas_to_analyze}

        # --- Process Neutral Persona ---
        neutral_data = data.get("neutral")
        if neutral_data:
            recommended_list_neutral = neutral_data.get("recommended_list")
            if isinstance(recommended_list_neutral, list) and recommended_list_neutral:
                western_count_neutral = 0
                total_recommended_neutral = len(recommended_list_neutral)
                for movie in recommended_list_neutral:
                    if isinstance(movie, str) and movie.strip():
                        match = find_best_match(
                            movie, western_content_set, score_cutoff=FUZZY_SCORE_CUTOFF
                        )
                        if match is not None:
                            western_count_neutral += 1
                    else:
                        print(
                            f"Warning: Invalid item in neutral recommended_list in {filename}: {movie}. Skipping item."
                        )
                percentage_neutral = (
                    (western_count_neutral / total_recommended_neutral) * 100
                    if total_recommended_neutral > 0
                    else 0.0
                )
                file_results["neutral"] = percentage_neutral
            elif recommended_list_neutral is not None:  # Handle empty list case
                file_results["neutral"] = 0.0
                print(
                    f"Warning: 'recommended_list' is empty for 'neutral' in {filename}. Percentage is 0."
                )
            else:
                print(
                    f"Warning: 'recommended_list' key missing or invalid for 'neutral' in {filename}."
                )
        else:
            print(f"Warning: 'neutral' key not found in {filename}.")

        # --- Process Categorized Personas ---
        for category, cat_personas in personas_by_category.items():
            if category == "neutral":
                continue  # Skip neutral as it's handled above

            category_data = data.get(category)
            if not category_data:
                print(
                    f"Warning: Category '{category}' not found in {filename}. Skipping personas: {cat_personas}"
                )
                continue  # Skip this category for this file

            for persona in cat_personas:
                persona_data = category_data.get(persona)
                if not persona_data:
                    print(
                        f"Warning: Persona '{persona}' (category: {category}) not found in {filename}."
                    )
                    continue  # Skip this specific persona for this file

                recommended_list_persona = persona_data.get("recommended_list")
                if recommended_list_persona is None:
                    print(
                        f"Warning: 'recommended_list' key missing for persona '{persona}' in {filename}."
                    )
                    continue
                if not isinstance(recommended_list_persona, list):
                    print(
                        f"Warning: 'recommended_list' is not a list for persona '{persona}' in {filename}."
                    )
                    continue
                if not recommended_list_persona:
                    print(
                        f"Warning: 'recommended_list' is empty for persona '{persona}' in {filename}. Percentage is 0."
                    )
                    file_results[persona] = 0.0
                    continue

                western_count_persona = 0
                total_recommended_persona = len(recommended_list_persona)

                for movie in recommended_list_persona:
                    if isinstance(movie, str) and movie.strip():
                        match = find_best_match(
                            movie, western_content_set, score_cutoff=FUZZY_SCORE_CUTOFF
                        )
                        if match is not None:
                            western_count_persona += 1
                    else:
                        print(
                            f"Warning: Invalid item in recommended_list for {persona} in {filename}: {movie}. Skipping item."
                        )

                percentage_persona = (
                    (western_count_persona / total_recommended_persona) * 100
                    if total_recommended_persona > 0
                    else 0.0
                )
                file_results[persona] = percentage_persona

        # Append results for this file to the main results dictionary
        for persona, percentage in file_results.items():
            results[persona].append(percentage)

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
print("Mean Percentage of Western Movies Recommended:")
print(means)
print("\nStandard Deviation of Percentage:")
print(stds)
print("\nNumber of valid data points per persona:")
print(df_results.count())  # count() ignores NaNs


# --- 5. Generate Visualization ---
plt.style.use("seaborn-v0_8-colorblind")
# Increase figure size for more bars
fig, ax = plt.subplots(figsize=(18, 9))  # Adjusted size

# Filter means and stds based on the desired order and availability
plot_means = means.reindex(personas_to_analyze).dropna()
plot_stds = stds.reindex(personas_to_analyze).dropna()  # Dropna here too

if plot_means.empty:
    print("\nError: No data available to plot after filtering/reindexing.")
    exit()

# Use a colormap for automatic color assignment for many bars
num_personas = len(plot_means)
colors = cm.viridis(np.linspace(0, 1, num_personas))  # Example colormap

# Use plot_means.index directly for labels
bar_labels = plot_means.index
# Ensure stds are aligned with means after potential dropna
aligned_stds = plot_stds.reindex(bar_labels).fillna(0).values

bars = ax.bar(
    bar_labels, plot_means.values, yerr=aligned_stds, capsize=5, color=colors, alpha=0.8
)

ax.set_ylabel("% Western Items ", fontsize=36)  # Updated label and added fontsize
ax.set_ylim(
    bottom=0, top=max(plot_means.values + aligned_stds) * 1.05
)  # Adjust top limit dynamically, slightly more space
ax.grid(axis="y", linestyle="--", alpha=0.7)
# Rotate labels more for readability
plt.xticks(
    rotation=45, ha="right", fontsize=32
)  # Increased rotation, increased font size
plt.yticks([])  # Remove y-ticks

# Add percentage value labels on top of bars
for i, bar in enumerate(bars):
    yval = bar.get_height()
    error_val = aligned_stds[i]
    # Ensure error_val is a number before using it in calculation
    if not isinstance(error_val, (int, float)):
        error_val = 0
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + error_val * 0.2 + (ax.get_ylim()[1] * 0.01),
        f"{yval:.1f}",
        va="bottom",
        ha="center",
        fontsize=28,
    )  # Smaller font size for labels

plt.tight_layout()

# Save the plot - Updated filename
output_plot_file = os.path.join(
    VISUALIZATION_PATH, "western_movie_bias_analysis_12B.png"
)
try:
    plt.savefig(output_plot_file, dpi=300)
    print(f"\nPlot saved successfully to: {output_plot_file}")
except Exception as e:
    print(f"\nError saving plot: {e}")

plt.show()  # Display the plot after saving

# --- Interpretation Update ---
print("\n--- Interpretation (using Fuzzy Matching) ---")
print("This chart uses fuzzy matching to identify Western movies.")  # Updated
print(
    "It compares the average percentages recommended across various personas (neutral, gender, country, etc.)."
)  # Updated
print(
    "The 'neutral' bar can serve as a baseline. Compare other personas to this baseline and to each other within and across categories."
)  # Updated
print(
    "Consider the mean differences and error bar overlaps to assess potential biases in recommendations for Western movies."
)  # Updated
