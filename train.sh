# CUDA_VISIBLE_DEVICES=1 python src/Supernet/train.py \
# --train-dir 'data/imagenet/train' \
# --val-dir 'data/imagenet/val' \
# --batch-size 2 \
# --val-batch-size 200


CUDA_VISIBLE_DEVICES=1 \
python src/Supernet/cifar_train.py \
--root 'data/cifar10' \
--batch-size 64 \
--val-batch-size 200