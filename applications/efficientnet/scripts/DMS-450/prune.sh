torchrun --nproc_per_node=8 --master_port 29112  timm_pruning.py ./data/imagenet_torch --model efficientnet_b4 -b 96 --sched step --epochs 4 \
--decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 \
--drop 0.5 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .048 \
--experiment 7b_to_45_low --pin-mem   --input-size 3 224 224 \
--target 0.296 --mutator_lr  2e-5 --target_scheduler root --loss_type log --loss_weight 1

