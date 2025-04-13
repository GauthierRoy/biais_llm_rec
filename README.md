# Project Name

## Setup Instructions

### 1. Create and Activate a Virtual Environment

Before running this project, it's recommended to use a virtual environment to manage dependencies (python 3.12 is recommended).

```sh
python3.12 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Run the following command to install the necessary packages:

```sh
pip install -e .
```

### 3. Data 
- Datasets used are located in `data/datasets`
- The dataset creation process can be reviewed, modified, or used as inspiration by exploring the Python scripts located in the `dataset_creation` directory.
> **Note:** To use the `dataset_creation` scripts, ensure that you create a `.config` file following the `.config_template` and update it with your personal credentials.

> **Warning:** The reproducibility of the dataset creation process cannot be guaranteed due to the nature of API requests and the evolving nature of data over time.


## Ollama

```sh
pip install -e .[ollama]
```

TODO

## Vllm

```sh
pip install -e .[vllm]
```

Launch the server


```sh
huggingface-cli login
```

```sh
python -m vllm.entrypoints.openai.api_server --model="[model]"
```
As an example of a model: `google/gemma-3-4b-it`

If it the model takes to much memory, you can reduce its context window like this:

```sh
python -m vllm.entrypoints.openai.api_server --model=google/gemma-3-12b-it --max-model-len 20000
```