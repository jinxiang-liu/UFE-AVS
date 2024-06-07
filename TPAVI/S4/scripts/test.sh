setting='S4'

# "resnet" or "pvt"
visual_backbone="resnet" 
# visual_backbone="pvt" 


# weights_path="weights/resnet_best.pth"
weights_path="weights/pvt_best.pth"



CUDA_VISIBLE_DEVICES=1 python test.py \
    --session_name ${setting}_${visual_backbone} \
    --visual_backbone ${visual_backbone} \
    --weights ${weights_path} \
    --test_batch_size 1 \
    --tpavi_stages 0 1 2 3 \
    --tpavi_va_flag \
    # --save_pred_mask 
    






