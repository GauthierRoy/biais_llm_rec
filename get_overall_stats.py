import json
import os


def calculate_metrics_stats(data: dict) -> dict:
    """
    Calculates overall mean and mean of standard deviations for divergence metrics.

    This function iterates through a dictionary of attributes, skipping the 'neutral' key.
    It collects both the 'mean' and 'std' values for 'IOU Divergence',
    'SERP MS Divergence', and 'Pragmatic Divergence', then calculates the
    overall average for each, rounded to two decimal places.

    Args:
        data: A dictionary of metric data loaded from a JSON file.

    Returns:
        A dictionary containing the calculated overall stats for each metric.
    """
    iou_means, iou_stds = [], []
    serp_means, serp_stds = [], []
    prag_means, prag_stds = [], []

    for attribute_name, attribute_data in data.items():
        if attribute_name == "neutral":
            continue

        if "IOU Divergence" in attribute_data:
            iou_means.append(attribute_data["IOU Divergence"].get("mean", 0.0))
            iou_stds.append(attribute_data["IOU Divergence"].get("std", 0.0))

        if "SERP MS Divergence" in attribute_data:
            serp_means.append(attribute_data["SERP MS Divergence"].get("mean", 0.0))
            serp_stds.append(attribute_data["SERP MS Divergence"].get("std", 0.0))

        if "Pragmatic Divergence" in attribute_data:
            prag_means.append(attribute_data["Pragmatic Divergence"].get("mean", 0.0))
            prag_stds.append(attribute_data["Pragmatic Divergence"].get("std", 0.0))

    def calculate_average(lst):
        return sum(lst) / len(lst) if lst else 0.0

    stats = {
        "overall_mean_iou": round(calculate_average(iou_means), 2),
        "overall_mean_std_iou": round(calculate_average(iou_stds), 2),
        "overall_mean_serp": round(calculate_average(serp_means), 2),
        "overall_mean_std_serp": round(calculate_average(serp_stds), 2),
        "overall_mean_prag": round(calculate_average(prag_means), 2),
        "overall_mean_std_prag": round(calculate_average(prag_stds), 2),
    }

    return stats


def process_and_print_stats(json_file_paths: list):
    """
    Processes a list of JSON files, calculates stats for each, and prints the results.

    Args:
        json_file_paths: A list of strings, where each string is a path to a JSON file.
    """
    if not json_file_paths:
        print("No file paths were provided.")
        return

    for file_path in json_file_paths:
        file_name = os.path.basename(file_path)
        print(f"\n--- Statistics for: {file_name} ---")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                stats = calculate_metrics_stats(data)

                print(json.dumps(stats, indent=4))

        except FileNotFoundError:
            print(f"  ERROR: File not found at '{file_path}'")
        except json.JSONDecodeError:
            print(
                f"  ERROR: Could not decode JSON from '{file_path}'. Check for formatting errors."
            )
        except Exception as e:
            print(f"  An unexpected error occurred: {e}")

        print("-" * (23 + len(file_name)))


if __name__ == "__main__":
    files_to_analyze = [
        "data/metric_results/gemma3_4b_college_.json",
        "data/metric_results/gemma3_4b_music_.json",
        "data/metric_results/gemma3_4b_movie_.json",
        "data/metric_results/llama3.2_3b_college_.json",
        "data/metric_results/llama3.2_3b_music_.json",
        "data/metric_results/llama3.2_3b_movie_.json",
        "data/metric_results/gemma3_1b_college_.json",
        "data/metric_results/gemma3_1b_music_.json",
        "data/metric_results/gemma3_1b_movie_.json",
        "data/metric_results/gemma3_12b_college_.json",
        "data/metric_results/gemma3_12b_music_.json",
        "data/metric_results/gemma3_12b_movie_.json",
    ]

    process_and_print_stats(files_to_analyze)
