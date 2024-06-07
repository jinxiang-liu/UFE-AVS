from easydict import EasyDict as edict
import yaml
import pdb

"""
default config
"""
cfg = edict()
cfg.BATCH_SIZE = 4
cfg.LAMBDA_1 = 50

##############################
# TRAIN
cfg.TRAIN = edict()
# TRAIN.SCHEDULER
cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "pretrained_backbones/vggish-10086976.pth"
cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = False
cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
# cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "pretrained_backbones/vggish_pca_params-970ea276.pth"
cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = False
cfg.TRAIN.PRETRAINED_RESNET50_PATH = "pretrained_backbones/resnet50-19c8e357.pth"
cfg.TRAIN.PRETRAINED_PVTV2_PATH = "pretrained_backbones/pvt_v2_b5.pth"

###############################
# DATA
cfg.DATA = edict()
cfg.DATA.ANNO_CSV = "meta_data/s4_meta_data.csv"
cfg.DATA.ANNO_CSV_SHUFFLE = "meta_data/s4_meta_data_shuffled.csv"
cfg.DATA.DIR_IMG = "/avsbench/Single-source/s4_data/visual_frames"
cfg.DATA.DIR_AUDIO_LOG_MEL = "/avsbench/Single-source/s4_data/audio_log_mel"
cfg.DATA.DIR_MASK = "/avsbench/Single-source/s4_data/gt_masks"
cfg.DATA.ANNO_CSV_SEMI = "meta_data/s4_semi_meta_data.csv"
cfg.DATA.DIR_FLOW_X = "/avsbench/Single-source/s4_data/S4_flows/flows_x"
cfg.DATA.DIR_FLOW_Y = "/avsbench/Single-source/s4_data/S4_flows/flows_y"
cfg.FLOW_TYPE = 'cnn'       
cfg.FLOW_RESNET_PRETRAIN = True
cfg.DATA.IMG_SIZE = (224, 224)


###############################



if __name__ == "__main__":
    print(cfg)
    pdb.set_trace()
