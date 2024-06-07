import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms
from copy import deepcopy

import json 
from .transforms_ops import *


def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach()  # [5, 1, 96, 64]
    return audio_log_mel

def load_flow(path, sizes=(224, 224)):
    return Image.open(path).convert('L').resize(sizes, Image.BICUBIC)

class MS3DatasetUFE(Dataset):
    """Dataset for multiple sound source segmentation"""

    def __init__(self, split='train', label=True, cfg=None):
        super(MS3DatasetUFE, self).__init__()
        self.size = 224
        self.label = label 
        self.split = split
        self.mask_num = 5
        self.cfg = cfg
        df_all = pd.read_csv(cfg.anno_csv_time, sep=',')
        with open(cfg.extended_file) as f:
            self.extended_file = json.load(f)
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split),
              len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

        self.flow_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])

    def __getitem__(self, index):
        if self.label or self.split == 'test':
            df_one_video = self.df_split.iloc[index]
            video_name = df_one_video[0]
            img_base_path = os.path.join(self.cfg.dir_img, video_name)
            audio_lm_path = os.path.join(
                self.cfg.dir_audio_log_mel, self.split, video_name + '.pkl')
            mask_base_path = os.path.join(
                self.cfg.dir_mask, self.split, video_name)
            audio_log_mel = load_audio_lm(audio_lm_path)
            flow_x_base_path = os.path.join(self.cfg.dir_flow_x, video_name)
            flow_y_base_path = os.path.join(self.cfg.dir_flow_y, video_name)
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        imgs, masks = [], []
        flow_xs, flow_ys = [], [] 
        if self.split == 'train':
            if self.label:
                time_id = int(df_one_video[2])
                img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s.mp4_%d.png" % (
                    video_name, time_id)))
                flow_x = load_flow(os.path.join(flow_x_base_path, "%s.mp4_%d.png"%(video_name, time_id)))
                flow_y = load_flow(os.path.join(flow_y_base_path, "%s.mp4_%d.png"%(video_name, time_id)))
                mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png" % (
                    video_name, time_id)), mode='P')
                img, mask, flow_x, flow_y = resize(img, mask, (0.5, 2.0), flowx=flow_x, flowy=flow_y)
                img, mask, flow_x, flow_y = crop(img, mask, self.size, flowx=flow_x, flowy=flow_y)
                img, mask, flow_x, flow_y = hflip(img, mask, p=0.5, flowx=flow_x, flowy=flow_y)
                img, mask, flow_x, flow_y = normalize(img, mask, high_res=False, flowx=flow_x, flowy=flow_y)
                audio_log_mel = audio_log_mel[time_id - 1]
                imgs.append(img)
                flow_xs.append(flow_x)
                flow_ys.append(flow_y)
                masks.append(mask)
                imgs_tensor = torch.stack(imgs, dim=0)
                masks_tensor = torch.stack(masks, dim=0)
                flow_x_tensor = torch.stack(flow_xs, dim=0)
                flow_y_tensor = torch.stack(flow_ys, dim=0)
                flow_tensor = torch.cat((flow_x_tensor, flow_y_tensor), dim=1)
                return imgs_tensor, audio_log_mel, masks_tensor, flow_tensor, video_name
            else:
                unlabel_path_prefix = "/remote-home/share/datasets/avsbench/Multi-sources/extended_flows"
                unlabel_img_path = f"{unlabel_path_prefix}/images/{self.extended_file[index]}.png"
                unlabel_flow_x_path = f"{unlabel_path_prefix}/flows_x/{self.extended_file[index]}/{self.extended_file[index]}.mp4_middle.png"
                unlabel_flow_y_path = f"{unlabel_path_prefix}/flows_y/{self.extended_file[index]}/{self.extended_file[index]}.mp4_middle.png"
                unlabel_audio_path = f"{unlabel_path_prefix}/audios_log_mel/{self.extended_file[index]}.pkl"
                img = load_image_in_PIL_to_Tensor(unlabel_img_path)
                audio_log_mel = load_audio_lm(unlabel_audio_path)
                flow_x = load_flow(unlabel_flow_x_path)
                flow_y = load_flow(unlabel_flow_y_path)
                mask = None 
                ignore_value = 254
                img, mask, flow_x, flow_y = resize(img, mask, (0.5, 2.0), flowx=flow_x, flowy=flow_y)
                img, mask, flow_x, flow_y = crop(img, mask, self.size, ignore_value, flowx=flow_x, flowy=flow_y)
                img, mask, flow_x, flow_y = hflip(img, mask, p=0.5, flowx=flow_x, flowy=flow_y)

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
                img_s1, ignore_mask, flow_x_s1_tensor, flow_y_s1_tensor = normalize(img_s1, ignore_mask, high_res=False, flowx=flow_x_s1, flowy=flow_y_s1)
                img_s2, _ = normalize(img_s2, high_res=False)
                mask = torch.from_numpy(np.array(mask)).long()
                ignore_mask = ignore_mask.squeeze(0)
                ignore_mask[mask == 254] = 255
                img_w_tensor, _, flow_x_w_tensor, flow_y_w_tensor = normalize(img_w, high_res=False, flowx=flow_x_w, flowy=flow_y_w)
                img_w_tensor = img_w_tensor
                img_s1_tensor = img_s1
                img_s2_tensor = img_s2
                ignore_mask_tensor = ignore_mask
                cutmix_box1_tensor = cutmix_box1
                cutmix_box2_tensor = cutmix_box2
                spectrogram = audio_log_mel[0]
                flow_w_tensor = torch.cat([flow_x_w_tensor, flow_y_w_tensor], dim=0)
                flow_s1_tensor = torch.cat([flow_x_s1_tensor, flow_y_s1_tensor], dim=0)
                return img_w_tensor, img_s1_tensor, img_s2_tensor, ignore_mask_tensor, cutmix_box1_tensor, cutmix_box2_tensor, spectrogram, flow_w_tensor, flow_s1_tensor
        else:
            for img_id in range(1, 6):
                img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s.mp4_%d.png" % (
                    video_name, img_id)), transform=self.img_transform)
                imgs.append(img)
                flow_x = self.flow_transform(load_flow(os.path.join(flow_x_base_path, "%s.mp4_%d.png"%(video_name, img_id))))
                flow_y = self.flow_transform(load_flow(os.path.join(flow_y_base_path, "%s.mp4_%d.png"%(video_name, img_id))))
                flow_xs.append(flow_x)
                flow_ys.append(flow_y)
            for mask_id in range(1, self.mask_num + 1):
                mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png" % (
                    video_name, mask_id)), transform=self.mask_transform, mode='P')
                masks.append(mask)
            imgs_tensor = torch.stack(imgs, dim=0)
            masks_tensor = torch.stack(masks, dim=0)
            flow_x_tensor = torch.stack(flow_xs, dim=0)
            flow_y_tensor = torch.stack(flow_ys, dim=0)
            flow_tensor = torch.cat((flow_x_tensor, flow_y_tensor), dim=1)

            return imgs_tensor, audio_log_mel, masks_tensor, flow_tensor, video_name

    def __len__(self):
        if self.split == 'train' and not self.label:
            return len(self.extended_file)
        return len(self.df_split)
