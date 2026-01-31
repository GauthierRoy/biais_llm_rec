#!/bin/bash
#SBATCH --job-name=vllm_inference
#SBATCH --output=/vast/projects/dyer1/lab/user/alex/biais_llm_rec/logs/%x_%j.out      # Standard output log (%x=job_name, %j=job_id)
#SBATCH --error=/vast/projects/dyer1/lab/user/alex/biais_llm_rec/logs/%x_%j.err       # Error log
#SBATCH --partition=dgx-b200         # Your specific partition
#SBATCH --gpus=1                     # Request 1 GPU
#SBATCH --cpus-per-gpu=16            # Your CPU request
#SBATCH --mem-per-gpu=128G           # Your Memory request
#SBATCH --time=03:00:00              # 3 Hour limit

cd /vast/projects/dyer1/lab/user/alex/biais_llm_rec
source .venv/bin/activate

echo "Date: $(date)"
echo "Host: $(hostname)"

bash pipeline.sh