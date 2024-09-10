## Fine-tuning LUT

The below run will fine-tune LUT on ImageNet. We used 8 V100s for fine-tuning:
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${num_gpus_per_node}  main_finetune.py \
        --accum_iter ${finetune_accum_iter} \
        --batch_size ${finetune_per_gpu_batch_size} \
        --model ${model_name}  \
        --finetune ${your_path}/${exp_name}/pretraining/checkpoint-${pretrain_epochs}.pth   \
        --output_dir ${your_path}/${exp_name}/finetune_seed${seed}  \
        --epochs 100 \
        --blr 1e-3 \
        --layer_decay 0.75 \
        --weight_decay 0.05 \
        --drop_path 0.1 \
        --mixup 0.8 \
        --cutmix 1.0 \
        --reprob 0.25 \
        --dist_eval \
        --data_path ${local_data_path}  \
        --save_periods last best \
        --auto_resume
```

The following commands provide recommended default settings:
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${num_gpus_per_node}  main_finetune.py \
        --accum_iter 4 \
        --batch_size 32 \
        --model vit_base_patch16  \
        --finetune ${your_path}/${exp_name}/pretraining/checkpoint-${pretrain_epochs}.pth   \
        --output_dir ${your_path}//${exp_name}/finetune  \
        --epochs 100 \
        --blr 1e-3 \
        --layer_decay 0.75 \
        --weight_decay 0.05 \
        --drop_path 0.1 \
        --mixup 0.8 \
        --cutmix 1.0 \
        --reprob 0.25 \
        --dist_eval \
        --data_path ${your_path}/data/ILSVRC2015/train/Data/CLS-LOC  \
        --save_periods last best \
        --auto_resume
```
