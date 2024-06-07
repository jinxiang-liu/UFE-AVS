
setting='S4'

## "resnet" or "pvt"
#visual_backbone="resnet"
visual_backbone="pvt" 

CUDA_VISIBLE_DEVICES=5 python train.py \
        --session_name ${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --train_batch_size 24\
        --max_epoches 120 \
        --lr 0.0001 \
        --tpavi_stages 0 1 2 3 \
        --tpavi_va_flag \
        --sa_loss_flag \
        --sa_loss_stages 0 1 2 3 \
        --num_workers 60 \
        --warmup_epoch 10 \
        --lambda_unsup 0.5


