import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
import random
import math
import cv2
from PIL import Image
from torchvision import transforms
from config import cfg
import pdb
import ipdb
from tqdm import tqdm

from dataset.transforms_ops import *


def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL

def load_flow(path, sizes=(224,224)):
    return Image.open(path).convert('L').resize(sizes, Image.BICUBIC)



def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach() # [5, 1, 96, 64]
    return audio_log_mel


class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, label=True, split='train', args=None):
        super(S4Dataset, self).__init__()
        self.label = label
        self.split = split
        self.args = args
        self.mask_num = 1 if self.split == 'train' else 5
        self.size = 224

       
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV_SEMI, sep=',')

        df_split = df_all[df_all['split'] == split]
        
        if split == "train":
            if label:
                df_split = df_split[df_split['label']=="Y"]
            else:
                df_split = df_split[df_split['label']=="N"]

       
        self.df_split = df_split
        

        if split == "train":
            print("{}/{} video frames are used for {}".format(len(self.df_split), len(df_all), self.split))
        else:
            print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df_split)


    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, category, time_id = df_one_video[0], df_one_video[2], df_one_video[4]
        
        
        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, self.split, category, video_name)
        flow_x_base_path = os.path.join(cfg.DATA.DIR_FLOW_X, self.split, category, video_name)
        flow_y_base_path = os.path.join(cfg.DATA.DIR_FLOW_Y, self.split, category, video_name)

        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, category, video_name)
        audio_lm_path = os.path.join(cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, category, video_name + '.pkl')
        audio_log_mel_5s = load_audio_lm(audio_lm_path)           # torch.Tensor: [5,1,96,64]

        imgs, masks, audio_log_mel = [], [], []
        flow_xs, flow_ys = [], []

        if self.split == 'train':
            img_id = int(time_id)
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id + 1)))
            flow_x = load_flow(os.path.join(flow_x_base_path, "%s_%d.png"%(video_name, img_id + 1)))
            flow_y = load_flow(os.path.join(flow_y_base_path, "%s_%d.png"%(video_name, img_id + 1)))

            audio_log_mel.append(audio_log_mel_5s[img_id])
            if self.label:
                mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, img_id + 1)), mode='1')
                ignore_value = 255
            else:
                mask = None
                ignore_value = 254

            img, mask, flow_x, flow_y = resize_with_flow(img, mask, flow_x, flow_y, (0.5, 2.0))
            img, mask, flow_x, flow_y = crop_with_flow(img, mask, flow_x, flow_y, self.size, ignore_value)
            img, mask, flow_x, flow_y = hflip_with_flow(img, mask, flow_x, flow_y, p=0.5)



            if self.label:
                img, mask, flow_x, flow_y = normalize_with_flow(img, mask, flow_x, flow_y)
                imgs.append(img)
                masks.append(mask)
                flow_xs.append(flow_x)
                flow_ys.append(flow_y)
                
                imgs_tensor = imgs[0]
                masks_tensor = masks[0]
                spectrogram = audio_log_mel[0]
                flow_x = flow_xs[0]
                flow_y = flow_ys[0]
                flow_tensor = torch.cat((flow_x, flow_y), dim=0)

                return imgs_tensor, spectrogram, masks_tensor, flow_tensor
            

            ### for unlabelled training data
            assert mask is None, "Mask should be Nones for unlabelled data"
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))

            img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

            flow_x_w, flow_y_w = deepcopy(flow_x), deepcopy(flow_y)
            flow_x_s1, flow_y_s1 = deepcopy(flow_x), deepcopy(flow_y)
            flow_x_s2, flow_y_s2 = deepcopy(flow_x), deepcopy(flow_y)
            
            if random.random() < 0.8:
                img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
            img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
            img_s1 = blur(img_s1, p=0.5)
            cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

            if random.random() < 0.8:
                img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
            img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
            img_s2 = blur(img_s2, p=0.5)
            cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

            ignore_mask = np.array(Image.fromarray(np.zeros((mask.size[1], mask.size[0]))))

            img_s1, ignore_mask, flow_x_s1_tensor, flow_y_s1_tensor = normalize_with_flow(img_s1, ignore_mask, flow_x_s1, flow_y_s1)
            img_s2, _ = normalize(img_s2)
            mask = torch.from_numpy(np.array(mask)).long()
            
            ignore_mask = ignore_mask.squeeze(0)
            ignore_mask[mask == 254] = 255

            img_w_tensor, _, flow_x_w_tensor, flow_y_w_tensor = normalize_with_flow(img_w, mask=None, flowx=flow_x_w, flowy=flow_y_w)
           
            img_w_tensor = img_w_tensor
            img_s1_tensor = img_s1
            img_s2_tensor = img_s2
            ignore_mask_tensor = ignore_mask
            cutmix_box1_tensor = cutmix_box1
            cutmix_box2_tensor = cutmix_box2
            spectrogram = audio_log_mel[0]

            flow_w_tensor = torch.cat([flow_x_w_tensor, flow_y_w_tensor], dim=0)
            flow_s1_tensor = torch.cat([flow_x_s1_tensor, flow_y_s1_tensor], dim=0) 

            return img_w_tensor, img_s1_tensor, img_s2_tensor, ignore_mask_tensor, \
                cutmix_box1_tensor, cutmix_box2_tensor, spectrogram, flow_w_tensor, flow_s1_tensor
        

        ### Validation or test
        else:
            for img_id in range(1, self.mask_num + 1):
                img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id )))
                mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, img_id)), mode='1')
                flow_x = load_flow(os.path.join(flow_x_base_path, "%s_%d.png"%(video_name, img_id)))
                flow_y = load_flow(os.path.join(flow_y_base_path, "%s_%d.png"%(video_name, img_id)))
                
                img, mask, flow_x, flow_y = normalize_with_flow(img, mask, flow_x, flow_y)

                imgs.append(img)
                masks.append(mask)
                flow_xs.append(flow_x)
                flow_ys.append(flow_y)
                audio_log_mel.append(audio_log_mel_5s[img_id-1])

            imgs_tensor = torch.stack(imgs, dim=0)     # torch.Size([5, 3, 224, 224])
            masks_tensor = torch.stack(masks, dim=0)    # torch.Size([5, 224, 224])  
            audio_log_mel = torch.stack(audio_log_mel, dim=0)   # torch.Size([5, ,1, 96, 64])
            
            flow_x_tensor = torch.stack(flow_xs, dim=0)
            flow_y_tensor = torch.stack(flow_ys, dim=0)
            flow_tensor = torch.cat((flow_x_tensor, flow_y_tensor), dim=1)

            return imgs_tensor, audio_log_mel, masks_tensor, flow_tensor, category, video_name



if __name__ == "__main__":
    test_dataset = S4Dataset('test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     num_workers=8,
                                                     pin_memory=True)

    with open('test_videos.txt', 'w') as f:
        for n_iter, batch_data in tqdm(enumerate(test_dataloader)):
            imgs, audio, mask, video_category, video_name = batch_data 
            f.writelines(video_category[0] + '/' +video_name[0] + '\n')
       
