echo "Running pipeline.sh"

# Manage virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping creation."
    source venv/bin/activate
else
    python3 -m venv venv
    source venv/bin/activate
    pip install .
fi


# Manage the dataset creation
mkdir -p data/datasets
dataset_list=()
for file in dataset_creation/*.py; do
    dataset_list+=("$(basename "${file%.py}")")
done

for dataset in "${dataset_list[@]}"; do
    if [ -f "data/datasets/$dataset.json" ]; then
        echo "$dataset dataset already exists, skipping creation."
    else
        echo "Creating dataset $dataset..."
        python "dataset_creation/${dataset}.py"  
        echo "$dataset dataset created."
    fi
done

# launch the inference 
python inference.py

# compute the metrics
python result.py

# create the visualization
python visualization.py