CONFIG=/MS3/config/AVSegFormer_pvt2_ms3_ufe.py
WEIGHTS=checkpoints/ms3_pvt2.pth

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=3 python scripts/$SESSION/test_ufe.py \
        $CONFIG \
        $WEIGHTS \
        # --save_pred_mask
