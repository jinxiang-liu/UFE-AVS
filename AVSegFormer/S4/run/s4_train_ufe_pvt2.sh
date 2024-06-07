CONFIG="S4/config/AVSegFormer_pvt2_s4_ufe.py"
OUTPUT_DIR="output/pvt2_ufe_s4"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python scripts/train_ufe.py $CONFIG --checkpoint_dir $OUTPUT_DIR --warmup_epoch 10
