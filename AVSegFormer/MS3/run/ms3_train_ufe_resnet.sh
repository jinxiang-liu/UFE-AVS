CONFIG="MS3/config/AVSegFormer_res50_ms3_ufe.py"
OUTPUT_DIR="output/resnet_ufe_ms3"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=2 python scripts/train_ufe.py $CONFIG --checkpoint_dir $OUTPUT_DIR --warmup_epoch 10
