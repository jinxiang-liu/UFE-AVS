import torch
import time
import torch.nn
import os
import random
import numpy as np
from mmengine.config import Config
import argparse
from utils import pyutils
from utils.loss_util import LossUtil
from utility import mask_iou
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset
from loss import IouSemanticAwareLoss


def main():
    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # logger
    log_name = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, dir_name)):
        os.mkdir(os.path.join(args.log_dir, dir_name))
    log_file = os.path.join(args.log_dir, dir_name, f'{log_name}.log')
    logger = getLogger(log_file, __name__)
    logger.info(f'Load config from {args.cfg}')

    # config
    cfg = Config.fromfile(args.cfg)
    logger.info(cfg.pretty_text)
    checkpoint_dir = os.path.join(args.checkpoint_dir, dir_name)

    # model
    model = build_model(**cfg.model)
    if args.pretrained_checkpoint != None:
        model.load_state_dict(torch.load(args.pretrained_checkpoint))
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    logger.info("Total params: %.2fM" % (sum(p.numel()
                for p in model.parameters()) / 1e6))

    # dataset
    train_dataset_label = build_dataset(**cfg.dataset.train_label)
    train_dataloader_label = torch.utils.data.DataLoader(train_dataset_label,
                                                   batch_size=cfg.dataset.train_label.batch_size,
                                                   shuffle=True,
                                                   num_workers=cfg.process.num_works,
                                                   pin_memory=True)
    max_step = (len(train_dataset_label) // cfg.dataset.train_label.batch_size) * \
        cfg.process.train_epochs
    train_dataset_unlabel = build_dataset(**cfg.dataset.train_unlabel)
    train_dataloader_unlabel = torch.utils.data.DataLoader(train_dataset_unlabel,
                                                           batch_size=cfg.dataset.train_unlabel.batch_size,
                                                           shuffle=True,
                                                           num_workers=cfg.process.num_works,
                                                           pin_memory=True)
    val_dataset = build_dataset(**cfg.dataset.val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=cfg.dataset.val.batch_size,
                                                 shuffle=False,
                                                 num_workers=cfg.process.num_works,
                                                 pin_memory=True)

    # optimizer
    optimizer = pyutils.get_optimizer(model, cfg.optimizer)
    loss_util = LossUtil(**cfg.loss)
    avg_meter_miou = pyutils.AverageMeter('miou')

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    for epoch in range(cfg.process.train_epochs):
        if epoch == cfg.process.freeze_epochs:
            model.module.freeze_backbone(False)

        loader = zip(train_dataloader_label, train_dataloader_unlabel, train_dataloader_unlabel)

        for n_iter, ((imgs, audio, mask, flows, _),
                     (img_u_w, img_u_s, _, ignore_mask, cutmix_box, _, spec_u1, flow_u_w, flow_u_s),
                     (img_u_w_mix, img_u_s_mix, _, ignore_mask_mix, _, _, spec_u2, _, flow_u_s_mix)) in enumerate(loader):

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            flows = flows.cuda()
            flow_u_w, flow_u_s, flow_u_s_mix = flow_u_w.cuda(), flow_u_s.cuda(), flow_u_s_mix.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B * frame, C, H, W)
            flows = flows.view(B * frame, 2, H, W)
            # mask_num = 5
            mask_num = 1
            mask = mask.view(B * mask_num, 1, H, W)
            # audio = audio.view(-1, audio.shape[2],
            #                    audio.shape[3], audio.shape[4])

            ignore_mask, ignore_mask_mix = ignore_mask.cuda(), ignore_mask_mix.cuda()
            img_u_w, img_u_w_mix = img_u_w.cuda(), img_u_w_mix.cuda()
            img_u_s, img_u_s_mix = img_u_s.cuda(), img_u_s_mix.cuda()
            cutmix_box = cutmix_box.cuda()
            spec_u1, spec_u2 = spec_u1.cuda(), spec_u2.cuda()

            model.eval()
            with torch.no_grad():
                pred_u_w_mix, _ = model(spec_u2, img_u_w_mix, flow_u_w)
                mask_u_w_mix = pred_u_w_mix.squeeze(dim=1)
                mask_u_w_mix = torch.sigmoid(mask_u_w_mix)
            
            img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = \
                img_u_s_mix[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]
            flow_u_s[cutmix_box.unsqueeze(1).expand(flow_u_s.shape) == 1] = \
                flow_u_s_mix[cutmix_box.unsqueeze(1).expand(flow_u_s.shape) == 1]
            
            model.train()
            output, mask_feature = model(audio, imgs, flows)
            pred_u_w, mask_feature_u_w = model(spec_u1, img_u_w, flow_u_w)
            pred_u_s, mask_feature_u_s = model(spec_u1, img_u_s, flow_u_s)

            mask_u_w_cutmixed, ignore_mask_cutmixed = \
                pred_u_w.detach().clone().squeeze(dim=1), ignore_mask.clone().squeeze(dim=1)
            mask_u_w_cutmixed = torch.sigmoid(mask_u_w_cutmixed)
            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w_mix[cutmix_box == 1]
            ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask_mix[cutmix_box == 1]

            loss, loss_dict = IouSemanticAwareLoss(
                output, mask_feature, mask, **cfg.loss)
            loss_u_s, loss_dict_u_s = IouSemanticAwareLoss(
                pred_u_s, mask_feature_u_s, mask_u_w_cutmixed.unsqueeze(1).unsqueeze(1), **cfg.loss)
            loss_u_s = loss_u_s * (ignore_mask_cutmixed != 255)
            loss_u_s = loss_u_s.sum() / (ignore_mask_cutmixed != 255).sum().item()
            if epoch > args.warmup_epoch:
                loss += args.lambda_unsup * loss_u_s 
            loss_util.add_loss(loss, loss_dict)
            loss_util.add_loss(loss_u_s, loss_dict_u_s)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if (global_step - 1) % 20 == 0:
                train_log = 'Iter:%5d/%5d, %slr: %.6f' % (
                    global_step - 1, max_step, loss_util.pretty_out(), optimizer.param_groups[0]['lr'])
                logger.info(train_log)

        # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]
                imgs, audio, mask, flows, _ = batch_data

                imgs = imgs.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                flows = flows.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B * frame, C, H, W)
                mask = mask.view(B * frame, H, W)
                flows = flows.view(B * frame, 2, H, W)
                audio = audio.view(-1, audio.shape[2],
                                   audio.shape[3], audio.shape[4])

                # [bs*5, 1, 224, 224]
                output, _ = model(audio, imgs, flows)

                miou = mask_iou(output.squeeze(1), mask)
                avg_meter_miou.add({'miou': miou})

            miou = (avg_meter_miou.pop('miou'))
            if miou > max_miou:
                model_save_path = os.path.join(
                    checkpoint_dir, '%s_best.pth' % (args.session_name))
                if not os.path.exists(model_save_path):
                    dir_path = os.path.dirname(model_save_path)
                    os.makedirs(dir_path)
                torch.save(model.module.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s' % model_save_path)

            miou_list.append(miou)
            max_miou = max(miou_list)

            val_log = 'Epoch: {}, Miou: {}, maxMiou: {}'.format(
                epoch, miou, max_miou)
            logger.info(val_log)

        model.train()
    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='config file path')
    parser.add_argument('--log_dir', type=str,
                        default='work_dir', help='log dir')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='work_dir', help='dir to save checkpoints')
    parser.add_argument("--session_name", default="MS3",
                        type=str, help="the MS3 setting")
    parser.add_argument("--warmup_epoch", default=20, type=int)
    parser.add_argument("--lambda_unsup", default=0.5, type=float)
    parser.add_argument("--pretrained_checkpoint", default=None)

    args = parser.parse_args()
    main()
