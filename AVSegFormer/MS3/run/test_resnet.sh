CONFIG=MS3/config/AVSegFormer_res50_ms3_ufe.py
WEIGHTS=checkpoints/ms3_resnet.pth

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=3 python scripts/test_ufe.py \
        $CONFIG \
        $WEIGHTS \
        # --save_pred_mask
