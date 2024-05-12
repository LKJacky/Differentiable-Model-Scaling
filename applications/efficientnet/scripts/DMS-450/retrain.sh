torchrun --nproc_per_node=8 --master_port 29112 \
timm_retrain.py ./data/imagenet_torch --model efficientnet_b4 -b 96 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .048 \
--experiment retain_45 --pin-mem --resume output/train/retain_45/last.pth.tar  \
--pruned checkpoints/450/pruned.pth --input-size 3 224 224 --target 0.2924 --teacher timm/efficientnet_b4 --teacher_input_image_size 320

