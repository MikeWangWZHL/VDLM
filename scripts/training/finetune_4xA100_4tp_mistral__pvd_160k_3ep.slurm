#!/bin/bash
#SBATCH --job-name="finetune_4xA100_4tp__pvd"
#SBATCH --output="logs/%j.%N.finetune_4xA100_4tp.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=64   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch&projects"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account="your account" # TODO: change to your account
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 18:00:00

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata

echo "job is starting on `hostname`"

# run the container binary with arguments: python3 <program.py>
# --bind /projects/bbXX  # add to apptainer arguments to mount directory inside container
WORK_DIR=`pwd`

# TODO: set the following to the path to the docker image "pt-megatron-llm_v1.1.sif"
IMAGE="path to the docker image" 
echo "WORK_DIR=$WORK_DIR"
echo "IMAGE=$IMAGE"

# TODO:
WANDB_API_KEY="your wandb api key"
# TODO:
HUGGING_FACE_HUB_TOKEN="your hugging face hub token"

echo "WANDB_API_KEY=$WANDB_API_KEY"
echo "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"


SCRIPT_TO_RUN=$1
echo "SCRIPT_TO_RUN=$SCRIPT_TO_RUN"


# check if the script to run exists
if [ ! -f "$SCRIPT_TO_RUN" ]; then
    echo "Script $SCRIPT_TO_RUN does not exist"
    exit 1
fi


apptainer run --nv \
    --no-home \
    --no-mount bind-paths \
    --cleanenv \
    --env "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --writable-tmpfs \
    --bind $WORK_DIR:/workspace \
    $IMAGE \
    /bin/bash -c "cd /workspace && $SCRIPT_TO_RUN"

# sbatch scripts/slurm/configs/finetune_4xA100_4tp.slurm
# squeue -u $USER
