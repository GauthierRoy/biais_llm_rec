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
pip install -r requirements.txt
```

### 3. Data 
- Datasets used are located in `data/datasets`
- The dataset creation process can be reviewed, modified, or used as inspiration by exploring the Python scripts located in the `dataset_creation` directory.
> **Note:** To use the `dataset_creation` scripts, ensure that the `.config` file is updated with your personal credentials.

> **Warning:** The reproducibility of the dataset creation process cannot be guaranteed due to the nature of API requests and the evolving nature of data over time.
