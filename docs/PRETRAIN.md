## Pre-training LUT

The below run will pre-train LUT on ImageNet. We used 8 V100s for pre-training:
```
MP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${num_gpus_per_node} --nnodes=${WORLD_SIZE} \
        --node_rank=${RANK}  --master_addr=${MASTER_ADDR}  --master_port=${MASTER_PORT} --use_env main_lut.py \
        --data_path ${local_data_path} \
        --model ${mim_name}_${model_name} \
        --batch_size ${pretrain_per_gpu_batch_size} \
        --accum_iter ${pretrain_accum_iter} \
        --blr 1.5e-4 \
        --weight_decay 0.05 \
        --model-ema-decay ${ema_decay} \
        --norm_pix_loss \
        --warmup_epochs 40 \
        --print_freq 100 \
        --epochs ${pretrain_epochs} \
        --output_dir ${your_path}/${exp_name}/pretraining \
        --save_periods last best every_100_epochs \
        --log_dir ${your_path}/${exp_name}/pretraining \
        --model-ema \
        --w_bc ${w_bc} \
        --seed ${seed} \
        --depth_head ${depth_head} \
        --depth_pred ${depth_pred} \
        --head_mlp_dim ${head_mlp_dim} \
        --head_norm_layer ${head_norm_layer} \
        --head_act_layer ${head_act_layer} \
        --auto_resume
```

The following commands provide recommended default settings:
```
MP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${num_gpus_per_node} --nnodes=${WORLD_SIZE} \
        --node_rank=${RANK}  --master_addr=${MASTER_ADDR}  --master_port=${MASTER_PORT} --use_env main_lut.py \
        --data_path ${your_path}/data/ILSVRC2015/train/Data/CLS-LOC \
        --model lut_vit_base_patch16 \
        --batch_size 128 \
        --accum_iter 4 \
        --blr 1.5e-4 \
        --weight_decay 0.05 \
        --model-ema-decay 0.996 \
        --norm_pix_loss \
        --warmup_epochs 40 \
        --print_freq 100 \
        --epochs 1600 \
        --output_dir ${your_path}/${exp_name}/pretraining \
        --save_periods last best every_100_epochs \
        --log_dir ${your_path}/${exp_name}/pretraining \
        --model-ema \
        --w_bc 0.25 \
        --seed 0 \
        --depth_head 2 \
        --depth_pred 2 \
        --head_mlp_dim 4096 \
        --head_norm_layer BN \
        --head_act_layer ReLU \
        --auto_resume
```
