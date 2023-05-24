import os
import time
import subprocess

import pandas as pd 

slurm_template = """#!/bin/bash -e
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_output}/slurm_%A.out
#SBATCH --error={slurm_output}/slurm_%A.err
#SBATCH --gpus={num_gpus}
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=36G
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.haopt12@vinai.io
#SBATCH --ntasks=1

module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/client/vinai/users/hieunt91/envs/flow
cd /lustre/scratch/client/vinai/users/hieunt91/cnf_flow

export MASTER_PORT={master_port}
export WORLD_SIZE=1

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export PYTHONPATH=$(pwd):$PYTHONPATH

export MODEL_TYPE={model_type}
export EPOCH_ID={epoch}
export DATASET={dataset}
export EXP={exp}
export OUTPUT_LOG={output_log}

echo "----------------------------"
echo $MODEL_TYPE $EPOCH_ID $DATASET $EXP {method} {num_steps}
echo "----------------------------"

CUDA_VISIBLE_DEVICES={device} python test_classifier_guidance.py --exp $EXP \
    --dataset $DATASET --batch_size 100 --epoch_id $EPOCH_ID \
    --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
    --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 3 \
    --model_type $MODEL_TYPE \
    --num_classes 1000 --label_dim 1000 --label_dropout 0.1 \
    --method {method} --num_steps {num_steps} \
    --compute_fid --output_log $OUTPUT_LOG \
    --scale="{scale}" --cfg_scale="1." \
    --classifier_ckpt ../quandm7/cnf_flow/saved_info/classifier_guidance/imagenet_latent/exp_1/model_225.pth \
    --classifier_depth 3 \
    --classifier_width 256 \
    --classifier_attention_resolutions 8,4 \
    --classifier_image_size 32 \
    --master_port $MASTER_PORT --num_process_per_node {num_gpus} \
    # --use_karras_samplers \

"""

###### ARGS
model_type = "DiT-B/2" # or "DiT-L/2" or "adm"
dataset = "latent_imagenet_256"
exp = "laflo_imnet_f8_ditb2" # "laflo_imnet_f8"
BASE_PORT = 8016
num_gpus = 8
device = "0,1,2,3,4,5,6,7"

config = pd.DataFrame({
    "epochs": [800]*3,
    "num_steps": [0]*3,
    "methods": ['euler']*3,
    "scale": [0., -1., -1.5],
})
print(config)

###################################
slurm_file_path = f"/lustre/scratch/client/vinai/users/hieunt91/cnf_flow/slurm_scripts/{exp}/run_cls.sh"
slurm_output = f"/lustre/scratch/client/vinai/users/hieunt91/cnf_flow/slurm_scripts/{exp}/"
output_log = f"{slurm_output}/log_cls"
os.makedirs(slurm_output, exist_ok=True)
job_name = "test"

for idx, row in config.iterrows():
    # device = str(idx % 2)
    # slurm_file_path = f"/lustre/scratch/client/vinai/users/haopt12/cnf_flow/slurm_scripts/{exp}/run{device}.sh"
    slurm_command = slurm_template.format(
        job_name=job_name,
        model_type=model_type,
        dataset=dataset,
        exp=exp,
        epoch=row.epochs,
        master_port=str(BASE_PORT+idx),
        slurm_output=slurm_output,
        num_gpus=num_gpus,
        output_log=output_log,
        method=row.methods,
        num_steps=row.num_steps,
        device=device,
        scale=row.scale,
    )
    mode = "w" if idx == 0 else "a"
    with open(slurm_file_path, mode) as f:
        f.write(slurm_command)
print("Slurm script is saved at", slurm_file_path)

# print(f"Summited {slurm_file_path}")
# subprocess.run(['sbatch', slurm_file_path])
