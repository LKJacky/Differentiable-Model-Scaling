export SRUN_ARGS="-N 1"
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=29111
export GPUS=8
export GPUS_PER_NODE=8

export USE_CEPH=true
export USE_SlURM=true

# total batch size=384*2=768=96*8

sh timm_slurm_train.sh eval test \
timm_retrain.py ../data/imagenet_torch --model efficientnet_b4 -b 96 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .048 \
--experiment retain_45 --pin-mem --resume output/train/retain_45/last.pth.tar  \
--pruned output/train/7b_to_45/last.pth.tar --input-size 3 224 224 --target 0.2925


