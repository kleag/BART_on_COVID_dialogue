#!/bin/bash

#SBATCH --job-name=BART_on_COVID_dialogue
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=gael.de-chalendar@cea.fr

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# #SBATCH --partition=amd
# #SBATCH --partition=classicgpu
# #SBATCH --partition=gpu40g
# #SBATCH --partition=gpup100
# #SBATCH --partition=gpuv100
#SBATCH --partition=lasti
# #SBATCH --partition=gpu-test

#SBATCH --gres=gpu:6

#SBATCH --time=1-00:00:00
####SBATCH --time=0-00:30:00

#SBATCH --mem=50G

echo "Begin on machine: `hostname`"

set -o nounset
set -o errexit
set -o pipefail

TOTAL_NUM_UPDATES=10000
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=1024
UPDATE_FREQ=1
BART_PATH=bart.large/model.pt

fairseq-train preprocess_data/patient2doctor-bin \
    --max-update 10000 \
    --max-epoch 100 \
    --patience 10 \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --memory-efficient-fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --ddp-backend=no_c10d \
    --disable-validation \
    --valid-subset train \
    --log-format simple --no-epoch-checkpoints
