setting='MS3'
visual_backbone="pvt" # "resnet" or "pvt"

CUDA_VISIBLE_DEVICES=6 python train.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --max_epoches 120 \
        --train_batch_size 16 \
        --lr 0.0001 \
        --tpavi_stages 0 1 2 3 \
        --tpavi_va_flag \
        --masked_av_flag \
        --masked_av_stages 0 1 2 3 \
        --lambda_1 0.5 \
        --kl_flag \
        --warmup_epoch 20 