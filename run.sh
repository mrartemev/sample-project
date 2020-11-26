#!/bin/bash

python main.py \
data.data_path=/Users/mrartemev/Datasets/Celeba/ \
utils.device=cpu \
experiment.batch_size=2 \
utils.num_workers=0 \
experiment.epochs=2 \
utils.epoch_iters=10 \
utils.eval_interval=1 \
utils.log_iter_interval=5 \
model.img_size=64 \
utils.save_interval=1