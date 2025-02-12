#$ -pe smp 8            # Number of cores.
#$ -l h_vmem=11G        # Memory per core (max 11G per GPU).
#$ -l h_rt=240:0:0        # Requested runtime.
#$ -cwd                 # Change to current directory.
#$ -j y                 # Join output and error files.
#$ -o outputs/          # Change default output directory.
#$ -l gpu=1             # Request GPU usage.
#$ -l gpu_type=ampere   # GPU type
#$ -t 1-6               # Array job.
#$ -tc 6                # Concurrent jobs.
#$ -m bea               # Email beginning, end, and aborted.

module load python

# Activate virtualenv
source .venv/bin/activate

inputs=(
    "UNetDatasets/MRIPhantom/prostate_Combined"
    "imagesTr"
    "labelsTr"
    "/data/scratch/exx851/UNet/MRI/prostate_Combined"
    )

folds=(
    0
    1
    2
    3
    4
    'all'
    )

dataset_path=${inputs[$((0))]}
training_images_path=${inputs[$((1))]}
training_labels_path=${inputs[$((2))]}
save_path=${inputs[$((3))]}
# Check if SGE_TASK_ID is 6 and set fold to "all"
if [ "$SGE_TASK_ID" -eq 6 ]; then
    fold="all"
else
    fold=${folds[$((SGE_TASK_ID-1))]}
fi

python unet_training.py \
  -dp="$dataset_path" \
  -tip="$training_images_path" \
  -tlp="$training_labels_path" \
  -sp="$save_path" \
  -bs=32 \
  -e=1000 \
  -is=600 \
  -p=0 \
  -f=$fold