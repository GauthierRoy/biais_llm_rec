#  LLM Ranking Evaluation Framework

**A framework for running inference experiments to evaluate Large Language Models (LLMs) on ranking tasks, potentially focusing on fairness or bias analysis based on sensitive attributes.**

---

## Table of Contents

1.  [Overview](#overview)
2.  [Prerequisites](#prerequisites)
3.  [Setup Instructions](#setup-instructions)
    *   [Clone Repository](#1-clone-repository)
    *   [Create Virtual Environment](#2-create-and-activate-a-virtual-environment)
    *   [Install Base Dependencies](#3-install-base-dependencies)
4.  [Data](#data)
    *   [Dataset Location](#dataset-location)
    *   [Dataset Creation (Optional)](#dataset-creation-optional)
5.  [Model Backend Setup](#model-backend-setup)
    *   [Ollama](#ollama)
    *   [VLLM](#vllm)
6.  [Configuration](#configuration)
    *   [Model Configuration (`model_config.json`)](#model-configuration-model_configjson)
    *   [Sensitive Attributes (`sensitive_attributes.json`)](#sensitive-attributes-sensitive_attributesjson)
    *   [Experiment Parameters (`config_inference.ini`)](#experiment-parameters-config_inferenceini)
7.  [Running the Experiment](#running-the-experiment)
    *   [Option 1: Using the Pipeline Script (Recommended)](#option-1-using-the-pipeline-script-recommended)
    *   [Option 2: Manual Execution](#option-2-manual-execution)
8.  [Results and Outputs](#results-and-outputs)
9.  [Contributing](#contributing)

---

## Overview

This project provides a structured way to run experiments using different Large Language Models (like those served via Ollama or VLLM) to perform ranking tasks on various datasets. It allows configuration of models, datasets, sensitive attributes (for potential bias analysis), and other parameters to systematically evaluate model performance and behavior.

---

## Prerequisites

*   **Python:** 3.12 (recommended)
*   **Git:** For cloning the repository.
*   **(Optional) Ollama:** If using Ollama as the model backend. [Installation Guide](https://ollama.com/)
*   **(Optional) VLLM compatible hardware (GPU):** If using VLLM as the model backend.

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/GauthierRoy/biais_llm_rec
cd biais_llm_rec
```

### 2. Install Dependencies

#### Option A: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (automatically creates virtual environment)
uv sync

# Run scripts using uv
uv run python inference.py
```

#### Option B: Using pip with virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate
# On Windows use `venv\Scripts\activate`

pip install -e .
```

---

## Data

### Dataset Location

*   The datasets used for inference are located in the `data/datasets/` directory. Ensure your required datasets are present here before running experiments.

### Dataset Creation (Optional)

*   Scripts used to generate the datasets can be found in the `dataset_creation/` directory. You can review these for methodology or adapt them for your own needs.

> **Note:** To run the `dataset_creation` scripts, you must:
> 1.  Copy the `.config_template` file to `.config`.
> 2.  Update the `.config` file with your personal credentials (e.g., API keys).


---

## Model Backend Setup

You need to set up a backend to serve the LLMs for inference. Choose **one** of the following options (Ollama or VLLM) and install its specific dependencies.

### Ollama

1.  **Install Ollama:** Follow the official [Ollama installation guide](https://ollama.com/).
2.  **Install Python Client:**
    ```bash
    pip install -e .[ollama]
    ```
3.  **Run Ollama Model:** Ensure the desired Ollama model is running before starting the experiment (e.g., `ollama run llama3`).

### VLLM

1.  **Install VLLM Support:**
    ```bash
    pip install -e .[vllm]
    ```
2.  **Login to Hugging Face (if needed):** Required for downloading models from Hugging Face Hub.
    ```bash
    huggingface-cli login
    ```
3.  **Launch VLLM OpenAI-Compatible Server:** Run this command in a **separate terminal** and keep it running during the experiment. Replace `[model_identifier]` with the Hugging Face model ID you want to use (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`).

    ```bash
    python -m vllm.entrypoints.openai.api_server --model "[model_identifier]"
    ```
    *Example:*
    ```bash
    python -m vllm.entrypoints.openai.api_server --model "meta-llama/Meta-Llama-3-8B-Instruct"
    ```
    *Memory Optimization:* If the model requires too much memory, you can limit its context window using `--max-model-len`:
    ```bash
    python -m vllm.entrypoints.openai.api_server --model "google/gemma-3-4b-it" --max-model-len 20000
    ```

---

## Configuration

Experiment parameters, models, and attributes are defined in configuration files within the `config/` directory.

### Model Configuration (`model_config.json`)

*   Define the models you intend to use in your experiments here. Specify details needed to connect to them (e.g., API endpoints if needed, model names recognized by the backend).
    *Example structure might be:*
    ```json
    {
      "llama3_ollama": { "model_name": "llama3", "api_base": "http://localhost:11434/v1", "backend": "ollama" },
      "llama3_vllm": { "model_name": "meta-llama/Meta-Llama-3-8B-Instruct", "api_base": "http://localhost:8000/v1", "backend": "vllm" }
    }
    ```

### Sensitive Attributes (`sensitive_attributes.json`)

*   Specify the sensitive attributes relevant to your datasets and analysis goals (e.g., gender, ethnicity, age group). This file guides any fairness or bias evaluation steps.
    *Example structure might be:*
    ```json
    {
      "college": ["gender", "ethnicity"],
      "movie": ["genre_preference"],
      "music": ["age_group"]
    }
    ```

### Experiment Parameters (`config_inference.ini`)

*   Parametrize your experiment runs using this INI file.

    ```ini
    [parameters]
    # List of model keys (defined in model_config.json) to run experiments on
    models = llama3_vllm, llama3_ollama

    # Comma-separated list of dataset types (corresponding to subdirectories in data/datasets/)
    dataset_types = college, movie, music

    # Optional: Contextual information added to the prompt, specific to each dataset type.
    # Order must match dataset_types. Use 'None' if no context for a specific dataset.
    type_of_activities = student, action movie fan, rock fan
    # Example with no context for 'movie':
    # type_of_activities = student, None, rock fan

    # Number of items the LLM should rank in its response
    k = 20

    # Specify the model backend type being used for this run ('vllm' or 'ollama')
    # Ensure this matches the setup and running server/service.
    type_inf = vllm

    # Comma-separated list of random seeds for experiment reproducibility
    seeds = 0, 1, 2, 3, 4
    ```

---

## Running the Experiment

Ensure your chosen model backend (Ollama service or VLLM server) is running before starting.

### Option 1: Using the Pipeline Script (Recommended)

This script automates the entire process: running inference, processing results, and generating visualizations based on your configuration.

```bash
bash pipeline.sh
```

### Option 2: Manual Execution

If you prefer to run the steps individually (e.g., for debugging or custom workflows):

1.  **Run Inference:** Executes the core LLM ranking task based on the configuration.
    ```bash
    python inference.py
    ```
    *This script reads `config_inference.ini`, loads datasets and models, sends prompts to the LLM backend, and saves the raw outputs.*

2.  **Process Results:** Aggregates and processes the raw outputs from the inference step.
    ```bash
    python result.py
    ```
    *This script likely calculates metrics, formats data, and prepares it for visualization.*

3.  **Generate Visualizations:** Creates plots or other visualizations from the processed results.
    ```bash
    python visualization.py
    ```
    *This script uses the processed data to generate charts/graphs, saved to a results directory.*

---

## Results and Outputs

*   **Raw Outputs:** The direct responses from the LLM during the `inference.py` step are typically saved in a structured format within a `results/raw/` directory 
*   **Processed Results:** Aggregated metrics and processed data generated by `result.py` are often stored in a `results/processed/` directory. 
*   **Visualizations:** Plots and tables created by `visualization.py` are usually saved as image files or interactive HTML files in a `results/visualizations/` or `results/plots/` directory.

--- 

## Contributing

We welcome contributions! If you have suggestions, improvements, or bug fixes, please create a pull request or open an issue in the repository. The original authors are:
- Alexandre André
- Gauthier Roy
