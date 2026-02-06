import configparser
import glob
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.lines import Line2D

# Replace with your actual imports
from pypalettes import load_cmap
from tqdm import tqdm

from utils.utils import find_best_match, get_correct_file_name

try:
    from utils.utils import find_best_match
except ImportError:
    import difflib

    def find_best_match(query, choices, score_cutoff=89):
        if not choices:
            return False
        matches = difflib.get_close_matches(
            query, choices, n=1, cutoff=score_cutoff / 100
        )
        return len(matches) > 0


# --- Global plot style (shared across horizontal figures) ---
PLOT_STYLE = "seaborn-v0_8-white"
PLOT_BASE_FONT = 9
PLOT_TITLE_FONT = 10
PLOT_TICK_FONT = 8
PLOT_LABEL_WEIGHT = "bold"
PLOT_LEGEND_WEIGHT = "bold"
PLOT_LEGEND_FONT = PLOT_TICK_FONT
PLOT_ANNOTATION_WEIGHT = "normal"
PLOT_FIG_WIDTH = 3.6
PLOT_MIN_HEIGHT = 2
PLOT_BAR_HEIGHT = 0.6
PLOT_ROW_GAP = 1.0


ROOT_PATH = ""
URL = "https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&release_date.lte=01%2F01%2F2024&sort_by=popularity.desc&vote_average.gte=8"


with open(f"{ROOT_PATH}.config", "r") as f:
    lines = f.readlines()
for line in lines:
    if line.startswith("TMBD_BEARER_TOKEN"):
        token = line.split("=")[1].strip()

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {token}",
}


def get_movies_by_country(title_list, target_countries):
    """
    Filters titles based on production countries (e.g., ['US', 'KR', 'FR']).
    """
    results = []
    search_url = "https://api.themoviedb.org/3/search/movie"
    detail_url = "https://api.themoviedb.org/3/movie/"  # Template for the {movie_id}

    # Standardize target codes to uppercase
    targets = set(c.upper() for c in target_countries)

    for title in tqdm(title_list):
        try:
            # Step 1: Search for the movie to get its TMDB ID
            search_params = {"query": title}
            search_resp = requests.get(
                search_url, headers=headers, params=search_params
            )

            if search_resp.status_code == 200:
                search_data = search_resp.json()
                if search_data["results"]:
                    # Take the best match
                    movie_id = search_data["results"][0]["id"]

                    # Step 2: Use the ID to get detailed movie info
                    # This is the endpoint you asked for: /movie/{movie_id}
                    full_details_resp = requests.get(
                        f"{detail_url}{movie_id}", headers=headers
                    )

                    if full_details_resp.status_code == 200:
                        movie_data = full_details_resp.json()

                        # Extract ISO codes from the production_countries list
                        # Structure is: [{'iso_3166_1': 'US', 'name': 'USA'}, ...]
                        prod_countries = [
                            c["iso_3166_1"]
                            for c in movie_data.get("production_countries", [])
                        ]

                        # Check if any of the production countries match your targets
                        if any(country in targets for country in prod_countries):
                            results.append(title)

        except Exception as e:
            print(f"Error processing {title}: {e}")

        # Small delay to stay within rate limits (now making 2 calls per title)
        time.sleep(0.001)

    return results


def get_genre_mapping():
    genre_url = "https://api.themoviedb.org/3/genre/movie/list?language=en"
    res = requests.get(genre_url, headers=headers)
    return {g["id"]: g["name"] for g in res.json().get("genres", [])}


def get_genres_for_titles(title_list, genres=["Action", "War"]):
    genre_map = get_genre_mapping()
    results = []

    # Convert input genres to a set for O(1) lookup and handle casing
    target_genres = set(g.capitalize() for g in genres)
    search_url = "https://api.themoviedb.org/3/search/movie"

    for title in tqdm(title_list):
        params = {"query": title, "language": "en-US"}
        try:
            response = requests.get(search_url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if data["results"]:
                    best_match = data["results"][0]
                    # Get actual names from IDs
                    movie_genres = set(
                        genre_map.get(gid) for gid in best_match.get("genre_ids", [])
                    )

                    # Check if there is ANY overlap between target_genres and movie_genres
                    if not target_genres.isdisjoint(movie_genres):
                        results.append(title)
        except Exception as e:
            print(f"Error searching for {title}: {e}")

        time.sleep(0.001)

    return results


def analyze_western_bias_horizontal(
    model_name,
    dataset_type,
    user_persona_type,
    label,
    movie_set,
    personas_config,
    fuzzy_cutoff=89,
    save=False,
    save_basename=None,
):
    # --- 1. Path Setup ---
    config = configparser.ConfigParser()
    config.read("config/config_inference")
    OUTPUT_PATH = config.get("paths", "output_path", fallback="results/")
    visualization_path = config.get("paths", "visualization_path", fallback="plots/")
    os.makedirs(visualization_path, exist_ok=True)

    all_persona_names = [p for sublist in personas_config.values() for p in sublist]
    results = {p_name: [] for p_name in all_persona_names}
    file_list = glob.glob(
        os.path.join(
            OUTPUT_PATH, f"{model_name}_{dataset_type}_{user_persona_type}_*.json"
        )
    )

    if not file_list:
        return None

    # --- 2. Data Processing ---
    for filename in file_list:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            for category, persona_list in personas_config.items():
                cat_data = data if category == "neutral" else data.get(category, {})
                if category == "location":
                    cat_data.update(data.get("country", {}))
                    cat_data.update(data.get("continent", {}))

                for p_name in persona_list:
                    p_data = (
                        cat_data.get(p_name)
                        if category != "neutral"
                        else data.get("neutral")
                    )
                    if p_data and "recommended_list" in p_data:
                        rec = p_data["recommended_list"]
                        matches = sum(
                            1
                            for item in rec
                            if find_best_match(
                                str(item), movie_set, score_cutoff=fuzzy_cutoff
                            )
                        )
                        results[p_name].append(
                            (matches / len(rec)) * 100 if rec else 0.0
                        )
        except Exception:
            continue

    # --- 3. Sorting & Layout Optimization ---
    df_results = pd.DataFrame(results)
    df_results.columns = [
        col.replace("an ", "").replace("a ", "").title() for col in df_results.columns
    ]
    means, stds = df_results.mean(), df_results.std().fillna(0)

    p_to_cat = {
        p.replace("an ", "").replace("a ", "").title(): c
        for c, plist in personas_config.items()
        for p in plist
    }

    plot_df = pd.DataFrame(
        {
            "Persona": means.index,
            "Mean": means.values,
            "Std": stds.values,
            "Category": [p_to_cat[p] for p in means.index],
        }
    )

    # Priority Sort: Neutral at top, then Mean Descending
    plot_df["Sort_Priority"] = np.where(plot_df["Persona"] == "Neutral", 0, 1)
    plot_df = plot_df.sort_values(
        by=["Sort_Priority", "Mean"], ascending=[True, False]
    ).reset_index(drop=True)
    plot_df = plot_df.iloc[::-1]  # Reverse for barh

    # --- 4. Column-Friendly Styling ---
    plt.style.use(PLOT_STYLE)
    fig_width = PLOT_FIG_WIDTH
    fig_height = max(PLOT_MIN_HEIGHT, len(plot_df) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Color Patch
    cmap = load_cmap("JeffKoons")
    raw_colors = (
        cmap.colors
        if hasattr(cmap, "colors")
        else [cmap(i) for i in np.linspace(0, 1, 10)]
    )
    manual_indices = {"neutral": 2, "location": 0, "race": 1, "religion": 3}
    cat_dict = {
        cat: sns.desaturate(raw_colors[manual_indices.get(cat, 0)], 0.8)
        for cat in plot_df["Category"].unique()
    }

    # Plot
    ax.barh(
        plot_df["Persona"],
        plot_df["Mean"],
        xerr=plot_df["Std"],
        color=[cat_dict[c] for c in plot_df["Category"]],
        height=PLOT_BAR_HEIGHT,
        linewidth=0.3,
        error_kw={"ecolor": "#333333", "elinewidth": 1, "capsize": 2, "capthick": 1},
    )
    # Reduce top/bottom padding so the first/last bars sit closer to the frame.
    ax.margins(y=0.02)

    # Clean Up
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100%"], fontsize=PLOT_TICK_FONT)
    ax.set_yticklabels(plot_df["Persona"], fontsize=max(6, PLOT_BASE_FONT - 2))

    ax.set_xlabel(
        f"Ratio of {label} Content",
        fontsize=PLOT_BASE_FONT,
        fontweight=PLOT_LABEL_WEIGHT,
    )
    # No title

    sns.despine(ax=ax, left=False, top=True, right=True)
    ax.xaxis.grid(True, linestyle=":", alpha=0.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#B0BEC5")
    ax.spines["bottom"].set_linewidth(1.0)

    # --- 5. Compact Legend (top) ---
    active_cats = [
        c for c in manual_indices.keys() if c in plot_df["Category"].unique()
    ]
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="w",
            marker="s",
            markersize=6,
            markerfacecolor=cat_dict[cat],
            label=cat.title(),
        )
        for cat in active_cats
    ]

    ax.legend(
        handles=legend_elements,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        prop={"weight": PLOT_LEGEND_WEIGHT, "size": PLOT_LEGEND_FONT},
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.3,
    )

    plt.tight_layout(pad=0.0)
    if save:
        base = (
            save_basename
            or f"{model_name}_{dataset_type}_{user_persona_type}_{label}_bias"
        )
        pdf_path = os.path.join(visualization_path, f"{base}.pdf")
        svg_path = os.path.join(visualization_path, f"{base}.svg")
        plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.0)
        plt.savefig(svg_path, bbox_inches="tight", pad_inches=0.0)
    plt.show()


def plot_persona_divergence_horizontal(
    models=("gemma3_12b",),
    dataset_types=("movie", "movie"),
    user_personas=("movie fan", "action movie fan"),
    metrics=("IOU Divergence", "Pragmatic Divergence", "SERP MS Divergence"),
    metric_index=0,
    desired_order=None,
    activity_color_map=None,
    config_path="config/config_inference",
    save=False,
    save_basename=None,
):
    """
    Horizontal (single-column) version of the persona divergence plot.
    Reads config/config_inference internally to get result_path and visualization_path.
    """

    if desired_order is None:
        desired_order = {
            "gender": ["a girl", "a boy", "a female", "a male"],
            "location": [
                "an American",
                "a Brazilian",
                "a British",
                "a Chinese",
                "a French",
                "a German",
                "a Japanese",
                "an African",
            ],
            "race": ["a Black", "a White", "an Asian", "a Hispanic"],
            "religion": ["a Buddhist", "a Christian", "a Muslim"],
        }

    cmap = load_cmap("JeffKoons")
    if activity_color_map is None:
        activity_color_map = {
            "movie fan": cmap.colors[0],
            "action movie fan": cmap.colors[3],
        }

    config = configparser.ConfigParser()
    config.read(config_path)

    result_path = config.get("paths", "result_path", fallback="results/")
    visualization_path = config.get("paths", "visualization_path", fallback="plots/")
    os.makedirs(visualization_path, exist_ok=True)

    metric = metrics[metric_index]
    categories = list(desired_order.keys())

    for model in models:
        print(f"Processing model: {model}...")
        df_activities = {}

        for dataset_type, user_persona in zip(dataset_types, user_personas):
            name_save = get_correct_file_name(f"{model}_{dataset_type}_{user_persona}")
            filepath = os.path.join(result_path, f"{name_save}.json")

            if not os.path.exists(filepath):
                print(f"Warning: File not found {filepath}")
                continue

            with open(filepath, "r") as f:
                final_metrics = json.load(f)

            final_metrics.pop("neutral", None)
            plot_data_list = []

            for attribute, metrics_dict in final_metrics.items():
                cleaned_attr = attribute.replace("an ", "").replace("a ", "").title()
                for metric_name, stats_dict in metrics_dict.items():
                    if metric_name == "mean_rank":
                        continue
                    plot_data_list.append(
                        {
                            "Attribute": cleaned_attr,
                            "Metric": metric_name,
                            "Mean": stats_dict["mean"],
                        }
                    )

            df_temp = pd.DataFrame(plot_data_list)
            if df_temp.empty:
                continue

            df_pivot = df_temp.pivot(index="Attribute", columns="Metric", values="Mean")
            df_activities[f"{user_persona}_mean"] = df_pivot

        if not df_activities:
            continue

        combined_df = pd.concat(
            df_activities, axis=0, names=["Activity", "Attribute"]
        ).reset_index()
        combined_df[["Activity Name", "Stat"]] = combined_df["Activity"].str.rsplit(
            "_", n=1, expand=True
        )
        combined_df = combined_df.pivot(
            index=["Attribute", "Activity Name"],
            columns="Stat",
            values=metric,
        ).reset_index()
        combined_df.rename(columns={"mean": f"Mean {metric}"}, inplace=True)

        def attr_to_category(attr):
            for cat, attrs in desired_order.items():
                cleaned = [
                    a.replace("an ", "").replace("a ", "").title() for a in attrs
                ]
                if attr in cleaned:
                    return cat
            return "other"

        combined_df["Category"] = combined_df["Attribute"].map(attr_to_category)

        wide = combined_df.pivot_table(
            index="Attribute",
            columns="Activity Name",
            values=f"Mean {metric}",
            aggfunc="mean",
        )

        order = []
        category_spans = []
        start_idx = 0
        for cat in categories:
            attrs = [
                a.replace("an ", "").replace("a ", "").title()
                for a in desired_order[cat]
            ]
            sub = wide.reindex(attrs)
            sorted_attrs = sub.max(axis=1).sort_values(ascending=False).index.tolist()
            order.extend(sorted_attrs)
            category_spans.append((start_idx, start_idx + len(sorted_attrs) - 1, cat))
            start_idx += len(sorted_attrs)

        wide = wide.reindex(order)

        # --- Plotting (horizontal) ---
        sns.set_style("white")

        # Tighter spacing to match analyze_western_bias_horizontal density
        row_gap = PLOT_ROW_GAP * 0.8
        fig_width = PLOT_FIG_WIDTH
        fig_height = max(6.0, len(order) * 0.30)
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

        for start, end, cat in category_spans:
            center = (start + end) / 2 * row_gap
            ax.text(
                1.02,
                center,
                cat.capitalize(),
                ha="left",
                va="center",
                transform=ax.get_yaxis_transform(),
                fontsize=PLOT_BASE_FONT,
                color="#000000",
                rotation=90,
                fontweight=PLOT_ANNOTATION_WEIGHT,
            )
            if start > 0:
                ax.axhline(
                    (start - 0.5) * row_gap, color="#000000", linewidth=1.0, zorder=1
                )

        wide["low_value"] = wide.min(axis=1)
        wide["high_value"] = wide.max(axis=1)
        low_mask = wide["movie fan"] <= wide["action movie fan"]
        wide["low_label"] = np.where(low_mask, "movie fan", "action movie fan")
        wide["high_label"] = np.where(low_mask, "action movie fan", "movie fan")

        bar_height = PLOT_BAR_HEIGHT * 0.8
        y_positions = np.arange(len(order)) * row_gap
        for label_type, z_order in [("high", 2), ("low", 3)]:
            val_col, lbl_col = f"{label_type}_value", f"{label_type}_label"
            for persona in ["movie fan", "action movie fan"]:
                mask = wide[lbl_col] == persona
                if not mask.any():
                    continue
                ax.barh(
                    y_positions[mask],
                    wide.loc[mask, val_col],
                    color=activity_color_map[persona],
                    height=bar_height,
                    linewidth=0.0,
                    zorder=z_order,
                )

        for y, x in zip(y_positions, wide["low_value"]):
            ax.vlines(
                x=x,
                ymin=y - 0.19 * row_gap,
                ymax=y + 0.19 * row_gap,
                color="#263238",
                linewidth=1.5,
                zorder=4,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5 * row_gap, (len(order) - 0.5) * row_gap)
        ax.set_xlabel(
            f"Mean {metric}",
            fontsize=PLOT_BASE_FONT,
            fontweight=PLOT_LABEL_WEIGHT,
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(order, fontsize=max(6, PLOT_BASE_FONT - 2))
        ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

        ax.tick_params(axis="y", length=4, width=0.8, color="#000000")
        ax.tick_params(
            axis="x",
            length=4,
            width=0.8,
            color="#000000",
            labelsize=PLOT_TICK_FONT,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#B0BEC5")
        ax.spines["bottom"].set_linewidth(1.0)

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=activity_color_map["action movie fan"]),
            plt.Rectangle((0, 0), 1, 1, color=activity_color_map["movie fan"]),
        ]
        labels = [
            "Action movie fan (w/ context)",
            "Movie fan (w/o context)",
        ]

        ax.legend(
            handles,
            labels,
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            prop={"weight": PLOT_LEGEND_WEIGHT, "size": PLOT_LEGEND_FONT},
            frameon=False,
            borderaxespad=0.1,
            columnspacing=0.8,
            handletextpad=0.3,
            handlelength=1.2,
            handleheight=1.2,
            labelspacing=0.8,
        )

        plt.tight_layout(pad=0.0)
        if save:
            base = (
                save_basename
                or f"{model}_{dataset_type}_{user_persona}_{metric}_divergence"
            )
            safe_base = base.replace(" ", "_")
            pdf_path = os.path.join(visualization_path, f"{safe_base}.pdf")
            svg_path = os.path.join(visualization_path, f"{safe_base}.svg")
            plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.0)
            plt.savefig(svg_path, bbox_inches="tight", pad_inches=0.0)
        plt.show()

    print("Processing complete.")


def plot_gender_bias_horizontal(
    model_name,
    dataset_type="movie",
    user_persona_type="movie",
    movie_set=None,
    config_path="config/config_inference",
    include_neutral=True,
    score_cutoff=89,
    save=False,
    save_basename=None,
):
    """
    Horizontal bar plot for gender-only personas (boy/male/girl/female [+ neutral]).
    Reads config/config_inference internally for output_path and visualization_path.
    Computes ratio of target movies in recommended_list using movie_set.
    """

    config = configparser.ConfigParser()
    config.read(config_path)

    output_path = config.get("paths", "output_path", fallback="results/")
    visualization_path = config.get("paths", "visualization_path", fallback="plots/")
    os.makedirs(visualization_path, exist_ok=True)

    if movie_set is None:
        raise ValueError(
            "movie_set is required (e.g., action_movies or romance_movies)."
        )

    file_pattern = os.path.join(
        output_path, f"{model_name}_{dataset_type}_{user_persona_type}_*.json"
    )
    file_list = glob.glob(file_pattern)

    if not file_list:
        print(f"No files found for: {file_pattern}")
        return None

    personas = ["a boy", "a male", "a girl", "a female"]
    if include_neutral:
        personas = ["neutral"] + personas

    results = {p: [] for p in personas}

    for filename in file_list:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Neutral
            if include_neutral:
                p_data = data.get("neutral")
                if p_data and "recommended_list" in p_data:
                    rec = p_data["recommended_list"]
                    matches = sum(
                        1
                        for item in rec
                        if find_best_match(
                            str(item), movie_set, score_cutoff=score_cutoff
                        )
                    )
                    results["neutral"].append(
                        (matches / len(rec)) * 100 if rec else 0.0
                    )

            # Gender: support both nested "gender" blocks and flat top-level keys
            gender_block = data.get("gender", {})
            for p in ["a boy", "a male", "a girl", "a female"]:
                p_data = gender_block.get(p) if gender_block else data.get(p)
                if p_data and "recommended_list" in p_data:
                    rec = p_data["recommended_list"]
                    matches = sum(
                        1
                        for item in rec
                        if find_best_match(
                            str(item), movie_set, score_cutoff=score_cutoff
                        )
                    )
                    results[p].append((matches / len(rec)) * 100 if rec else 0.0)

        except Exception:
            continue

    df = pd.DataFrame(results)
    df.columns = [c.replace("an ", "").replace("a ", "").title() for c in df.columns]

    means = df.mean(skipna=True)
    stds = df.std(skipna=True).fillna(0)

    order = [c.replace("an ", "").replace("a ", "").title() for c in personas]
    if include_neutral:
        if "Neutral" not in means:
            means.loc["Neutral"] = 0.0
            stds.loc["Neutral"] = 0.0

    # Style aligned with analyze_western_bias_horizontal
    plt.style.use(PLOT_STYLE)
    fig_width = PLOT_FIG_WIDTH
    fig_height = PLOT_MIN_HEIGHT
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Neutral color matches analyze_western_bias_horizontal (JeffKoons index 2 desaturated)
    cmap = load_cmap("JeffKoons")
    raw_colors = (
        cmap.colors
        if hasattr(cmap, "colors")
        else [cmap(i) for i in np.linspace(0, 1, 10)]
    )
    neutral_color = sns.desaturate(raw_colors[2], 0.8)

    if include_neutral:
        colors = [neutral_color, "skyblue", "steelblue", "lightcoral", "indianred"][
            : len(order)
        ]
    else:
        colors = ["skyblue", "steelblue", "lightcoral", "indianred"][: len(order)]

    order_plot = order[::-1] if include_neutral else order
    ax.barh(
        order_plot,
        means.reindex(order_plot),
        xerr=stds.reindex(order_plot),
        capsize=2,
        color=colors[::-1] if include_neutral else colors,
        height=PLOT_BAR_HEIGHT,
        linewidth=0.8,
        alpha=0.9,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )

    ax.grid(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    sns.despine(ax=ax, right=True, top=True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#B0BEC5")
    ax.spines["bottom"].set_linewidth(1.0)

    ax.set_xlabel(
        "Ratio of Target Movies (%)",
        fontsize=PLOT_BASE_FONT,
        fontweight=PLOT_LABEL_WEIGHT,
        labelpad=10,
    )
    # No title
    plt.xticks(fontsize=PLOT_TICK_FONT)
    plt.yticks(fontsize=max(6, PLOT_BASE_FONT - 2))
    max_val = (means.reindex(order_plot) + stds.reindex(order_plot)).max()
    max_val = float(max_val) if pd.notna(max_val) else 0.0
    x_max = max(1.0, max_val * 1.05)
    ax.set_xlim(0, x_max)

    if x_max > 50:
        step = 20
    elif x_max > 25:
        step = 10
    else:
        step = 5
    ticks = np.arange(0, x_max + 0.001, step)
    ax.set_xticks(ticks)

    plt.tight_layout(pad=0.0)
    if save:
        base = (
            save_basename
            or f"{model_name}_{dataset_type}_{user_persona_type}_gender_bias"
        )
        pdf_path = os.path.join(visualization_path, f"{base}.pdf")
        svg_path = os.path.join(visualization_path, f"{base}.svg")
        plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.0)
        plt.savefig(svg_path, bbox_inches="tight", pad_inches=0.0)
    plt.show()

    return df
