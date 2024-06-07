CONFIG=S4/config/AVSegFormer_pvt2_s4_ufe.py
WEIGHTS=checkpoints/s4_pvt2.pth
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python scripts/test_ufe.py \
        $CONFIG \
        $WEIGHTS \
        # --save_pred_mask
