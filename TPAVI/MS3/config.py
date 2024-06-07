from easydict import EasyDict as edict
import yaml
import pdb

"""
default config
"""
cfg = edict()
cfg.BATCH_SIZE = 4
cfg.LAMBDA_1 = 5
cfg.MASK_NUM = 5

##############################
# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "pretrained_backbones/vggish-10086976.pth"
cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = False
cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
# cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "/remote-home/share/yikunliu/avsbench/pretrained_backbones/vggish_pca_params-970ea276.pth"
cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = False
cfg.TRAIN.PRETRAINED_RESNET50_PATH = "pretrained_backbones/resnet50-19c8e357.pth"
cfg.TRAIN.PRETRAINED_PVTV2_PATH = "pretrained_backbones/pvt_v2_b5.pth"

###############################
# DATA
cfg.DATA = edict()
cfg.DATA.ANNO_CSV = "meta_data/ms3_meta_data.csv"
cfg.DATA.DIR_LABEL_IMG = "/avsbench/Multi-sources/ms3_data/visual_frames"
cfg.DATA.DIR_AUDIO_LABEL_LOG_MEL = "/avsbench/Multi-sources/ms3_data/audio_log_mel"
cfg.DATA.DIR_MASK = "/avsbench/Multi-sources/ms3_data/gt_masks"
# cfg.ANNO_CSV_TIME = "/avsbench/Multi-sources/ms3_time.csv"
cfg.EXTENDED_FILE = "avsbench/Multi-sources/MS3_Extend/data.json"
cfg.MS3_EXTEND_DIR="avsbench/Multi-sources/MS3_Extend/"
cfg.DIR_FLOW_X="avsbench/Multi-sources/MS3_flows//flows_x"
cfg.DIR_FLOW_Y="avsbench/Multi-sources/MS3_flows//flows_y"

cfg.FLOW_TYPE = 'cnn' 
cfg.DATA.IMG_SIZE = (224, 224)
###############################



if __name__ == "__main__":
    print(cfg)
    pdb.set_trace()
