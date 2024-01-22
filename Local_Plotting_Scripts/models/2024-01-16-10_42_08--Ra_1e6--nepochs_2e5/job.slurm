#!/bin/bash
#SBATCH --job-name=10:42:08
#SBATCH -p
#SBATCH gpu
#SBATCH --gpus=a100:1
#SBATCH --account=cmcdevitt-roy
#SBATCH --qos=cmcdevitt-roy
#SBATCH --mem-per-gpu=80gb
#SBATCH --time=120:00:00

module load cuda/11.1.0
module load conda
 conda activate env/
nvidia-smi > models/2024-01-16-10:42:08--Ra=1e6--nepochs=2e5/nvidia-output.txt

echo ''
echo -e 'Request Start Time:	'2024-01-16-10:42:08
CODE_START_TIME_UNIX=`date +\%s`
CODE_START_TIME_DATE=$(date "+%F-%H:%M:%S")
echo -e 'Code Start Time:   	'$CODE_START_TIME_DATE
echo ''

python  models/2024-01-16-10:42:08--Ra=1e6--nepochs=2e5/Script.py models/2024-01-16-10:42:08--Ra=1e6--nepochs=2e5/ $CODE_START_TIME_UNIX --Ra=1e6 --nepochs=2e5

echo -e 'Request Start Time:	'2024-01-16-10:42:08
echo -e 'Code Start Time:   	'$CODE_START_TIME_DATE
echo -e "Code End Time:     	"$(date "+%F-%H:%M:%S")
echo ''