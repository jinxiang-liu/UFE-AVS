import os
import time
import random
import shutil
import torch
import numpy as np
from einops import rearrange, repeat
import argparse
import logging

import torch.nn.functional as F
from config import cfg
from torchvggish import vggish
from loss import IouSemanticAwareLoss

from utils import pyutils
from utils.utility import logger, mask_iou
from utils.system import setup_logging
import ipdb

from tensorboardX import SummaryWriter
import utils.tensorboard_utils as TB

from utils.utils import AverageMeter


class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        with torch.no_grad():
            audio_fea = self.audio_backbone(audio)
        return audio_fea.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--session_name", default="S4", type=str, help="the S4 setting")
    parser.add_argument("--visual_backbone", default="resnet", type=str, help="use resnet50 or pvt-v2 as the visual backbone")
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=25, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument('--sa_loss_flag', action='store_true', default=False, help='additional loss for last four frames')
    parser.add_argument("--lambda_1", default=0.5, type=float, help='weight for balancing l4 loss')
    parser.add_argument("--lambda_2", default=0.2, type=float, help='weight for balancing l4 loss for weak and strong transformation')
    parser.add_argument("--lambda_unsup", default=0.5, type=float, help='weight for balancing supervised and semisupervised loss')
    parser.add_argument("--warmup_epoch", default=4, type=int, help="warm up epoch for fixmatch training")
    parser.add_argument('--aud_spec_aug', action='store_true', default=False, help='spec aug for log mel')
    parser.add_argument("--prob_logmel_perturb", default=0.5, type=float, help=' ')
    parser.add_argument("--logmel_perturb", default='w', type=str, help='logmel perturb strength: w s m  ')
    parser.add_argument("--sa_loss_stages", default=[], nargs='+', type=int, help='compute sa loss in which stages: [0, 1, 2, 3')
    parser.add_argument("--mask_pooling_type", default='avg', type=str, help='the manner to downsample predicted masks')
    parser.add_argument("--tpavi_stages", default=[], nargs='+', type=int, help='add tpavi block in which stages: [0, 1, 2, 3')
    parser.add_argument("--tpavi_vv_flag", action='store_true', default=False, help='visual-visual self-attention')
    parser.add_argument("--tpavi_va_flag", action='store_true', default=False, help='visual-audio cross-attention')
    parser.add_argument("--weights", type=str, default='', help='path of trained model')
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    parser.add_argument('--trainset_shuffle', default=False, action='store_true', help='')
    parser.add_argument("--trainset_ratio", type=float, default=1)

    args = parser.parse_args()

    if (args.visual_backbone).lower() == "resnet":
        from model import ResNet_AVSModel_flow as AVSModel
        print('==> Use ResNet50 as the visual backbone...')
    elif (args.visual_backbone).lower() == "pvt":
        from model import PVT_AVSModel_flow as AVSModel
        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")
    
    from dataloader import S4Dataset

    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime('%Y%m%d%H%M%S_' + prefix)))
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    scripts_to_save = [ ]
    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    # Model
    model = AVSModel.Pred_endecoder(channel=256, \
                                        config=cfg, \
                                        tpavi_stages=args.tpavi_stages, \
                                        tpavi_vv_flag=args.tpavi_vv_flag, \
                                        tpavi_va_flag=args.tpavi_va_flag)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    # Data
    train_dataset_u = S4Dataset(label=False, split='train', args=args)
    train_dataset_l = S4Dataset(label=True, split='train', args=args)

    train_dataloader_u = torch.utils.data.DataLoader(train_dataset_u,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)
    
    train_dataloader_l = torch.utils.data.DataLoader(train_dataset_l,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)


    max_step = min(len(train_dataloader_l), len(train_dataloader_u)) * args.max_epoches

    val_dataset = S4Dataset(split='val', args=args)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=args.val_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    # Optimizer
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, args.lr)
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_sup_loss = pyutils.AverageMeter('supervised_loss')
    avg_meter_unsup_loss = pyutils.AverageMeter('unsupervised_loss')
    avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')
    avg_meter_sa_loss = pyutils.AverageMeter('sa_loss')
    avg_meter_miou = pyutils.AverageMeter('miou')

    # Tensorboard
    writer_train = SummaryWriter(logdir=os.path.join(log_path, 'train'))
    # writer_val = SummaryWriter(logdir=os.path.join(log_path, 'val'))
    args.train_plotter = TB.PlotterThread(writer_train)
    # args.val_plotter = TB.PlotterThread(writer_val)


    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    # args.iteration = 1

    for epoch in range(args.max_epoches):
        losses = AverageMeter('Loss_total',':.4f')
        losses_sup = AverageMeter('Loss_sup',':.4f')
        losses_unsup = AverageMeter('Loss_unsup',':.4f')
        losses_iou = AverageMeter('Loss_iou',':.4f')
        losses_sa = AverageMeter('Loss_sa',':.4f')
        loader = zip(train_dataloader_l, train_dataloader_u, train_dataloader_u)

        for n_iter, ((img_x, spec_x, mask_x, flow_x),
                     (img_u_w, img_u_s, _, ignore_mask, cutmix_box, _, spec_u1, flow_u_w, flow_u_s),
                     (img_u_w_mix, img_u_s_mix, _, ignore_mask_mix, _, _, spec_u2, _, flow_u_s_mix)  ) in enumerate(loader):
            
            # lablelled data
            img_x, spec_x, mask_x, flow_x = img_x.cuda(), spec_x.cuda(), mask_x.cuda(), flow_x.cuda()
            bs_x = img_x.size(0)
            # mask_x: H, W
            
            # unlabelled data
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
            spec_u1 = spec_u1.cuda()
            spec_u2 = spec_u2.cuda()

            flow_u_w = flow_u_w.cuda()
            flow_u_s = flow_u_s.cuda()
            flow_u_s_mix = flow_u_s_mix.cuda()
            bs_un = img_u_w.size(0)    # batch size

            # Cut mix augmentation
            ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()
            img_u_w_mix, img_u_s_mix = img_u_w_mix.cuda(), img_u_s_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            # predict pseudo mask for unlalelled weak augmented images
            with torch.no_grad():
                model.eval()
                spec_u2_feature = audio_backbone(spec_u2)       # b*96*64->b*1*96*64 -> b*1*128
                pred_u_w_mix, _, _ = model(img_u_w_mix, spec_u2_feature, flow_u_w,1)
                mask_u_w_mix = pred_u_w_mix.squeeze(dim=1)   #.squeeze(dim=1)        # -> (b*1) h w
                mask_u_w_mix = torch.sigmoid(mask_u_w_mix)
                
            # cutout augmentation
            img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = \
                img_u_s_mix[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]
            
            # cutout for flow too
            flow_u_s[cutmix_box.unsqueeze(1).expand(flow_u_s.shape) == 1] = \
                flow_u_s_mix[cutmix_box.unsqueeze(1).expand(flow_u_s.shape) == 1]



            model.train()
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            spec_x_feature = audio_backbone(spec_x) 
            spec_img_u_w_feature = audio_backbone(spec_u1) 


            pred_x_and_uw, visual_map_list_x_and_uw, a_fea_list_x_and_uw = model(torch.cat((img_x, img_u_w)), \
                                                        torch.cat((spec_x_feature, spec_img_u_w_feature)), \
                                                        torch.cat((flow_x, flow_u_w)), \
                                                            1)
            pred_x, pred_u_w = pred_x_and_uw.split([num_lb, num_ulb])
            
            visual_map_list_x = [each.split([num_lb, num_ulb])[0] for each in visual_map_list_x_and_uw]
            visual_map_list_u_w = [each.split([num_lb, num_ulb])[1] for each in visual_map_list_x_and_uw]
            # a_fea_list_x, a_fea_list_uw = a_fea_list_x_and_uw.split([num_lb, num_ulb])
            a_fea_list_x = [ each.split([num_lb, num_ulb])[0] for each in a_fea_list_x_and_uw]
            a_fea_list_uw = [ each.split([num_lb, num_ulb])[1] for each in a_fea_list_x_and_uw]
            
            # predict with unlabelled strong augmented images
            pred_u_s, visual_map_list_u_s, a_fea_list_u_s  = model(img_u_s, spec_img_u_w_feature, flow_u_s, 1)

            mask_u_w_cutmixed, ignore_mask_cutmixed = \
                pred_u_w.detach().clone().squeeze(dim=1), ignore_mask.clone().squeeze(dim=1)
            
            mask_u_w_cutmixed = torch.sigmoid(mask_u_w_cutmixed)
            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w_mix[cutmix_box == 1]
            ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask_mix[cutmix_box == 1]

            # Labelled loss
            # pred_x: B*1*H*W   mask_x: B*H*W
            loss_x, loss_dict_x = IouSemanticAwareLoss(pred_x, mask_x.unsqueeze(1), \
                                                a_fea_list_x, visual_map_list_x, \
                                                lambda_1=args.lambda_1, \
                                                count_stages=args.sa_loss_stages, \
                                                sa_loss_flag=args.sa_loss_flag, \
                                                mask_pooling_type=args.mask_pooling_type
                                                )

            # Unlabelled consistency loss
            loss_u_s, loss_dict_u_s = IouSemanticAwareLoss(pred_u_s, mask_u_w_cutmixed.unsqueeze(1).unsqueeze(1),\
                                            a_fea_list_u_s, visual_map_list_u_s, \
                                            lambda_1=args.lambda_2, \
                                            count_stages=args.sa_loss_stages, \
                                            sa_loss_flag=args.sa_loss_flag, \
                                            mask_pooling_type=args.mask_pooling_type
                                            )
            
            loss_u_s = loss_u_s * (ignore_mask_cutmixed != 255)
            loss_u_s = loss_u_s.sum() / (ignore_mask_cutmixed != 255).sum().item()

            if epoch > args.warmup_epoch:
                loss = loss_x + args.lambda_unsup * loss_u_s
            else:
                loss = loss_x

            avg_meter_total_loss.add({'total_loss': loss.item()})
            avg_meter_sup_loss.add({'supervised_loss': loss_x.item()})
            avg_meter_unsup_loss.add({'unsupervised_loss': loss_u_s.item()})
            avg_meter_iou_loss.add({'iou_loss': loss_dict_x['iou_loss']})
            avg_meter_sa_loss.add({'sa_loss': loss_dict_x['sa_loss']})

            args.train_plotter.add_data('train/local/loss_total', loss.item(), global_step)
            args.train_plotter.add_data('train/local/loss_sup', loss_x.item(), global_step)
            args.train_plotter.add_data('train/local/loss_unsup', loss_u_s.item(), global_step)
            args.train_plotter.add_data('train/local/loss_iou', loss_dict_x['iou_loss'], global_step)
            args.train_plotter.add_data('train/local/loss_sa', loss_dict_x['sa_loss'], global_step)

            losses.update(loss.item(), bs_x + bs_un if epoch > args.warmup_epoch else bs_x)
            losses_sup.update(loss_x.item(), bs_x)
            losses_unsup.update(loss_u_s.item(), bs_un)
            losses_iou.update(loss_dict_x['iou_loss'], bs_x)
            losses_sa.update(loss_dict_x['sa_loss'], bs_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if (global_step-1) % 10 == 0:
                train_log = 'Epoch:%2d, Iter:%5d/%5d, Total_Loss:%.4f, Sup_Loss:%.4f, Unsup_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f,'%(
                            epoch, global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_sup_loss.pop("supervised_loss"), avg_meter_unsup_loss.pop("unsupervised_loss"), avg_meter_iou_loss.pop('iou_loss'), avg_meter_sa_loss.pop('sa_loss'), )
                
                logger.info(train_log)

                

        # 
        args.train_plotter.add_data('train/global/loss', losses.avg, epoch)
        args.train_plotter.add_data('train/global/loss_iou', losses_iou.avg, epoch)
        args.train_plotter.add_data('train/global/loss_sa', losses_sa.avg, epoch)

        # Validation:
        ious = AverageMeter()

        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                imgs, audio, mask, flows, _, _ = batch_data # [bs, 5, 3, 224, 224], [bs, T/5, 1, 96, 64], [bs, 5, 1, 224, 224]

                imgs = imgs.cuda()
                flows = flows.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                _, _, C_f, H, W = flows.shape
                imgs = imgs.view(B*frame, C, H, W)
                flows = flows.view(B*frame, C_f, H, W)
                
                mask = mask.view(B*frame, H, W)
                audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])  # [B*T/5, 128]
                with torch.no_grad():
                    audio_feature = audio_backbone(audio)
                output, _, _ = model.module(imgs, audio_feature, flows, 5) # [bs*5, 1, 224, 224]
                miou = mask_iou(output.squeeze(1), mask)
                avg_meter_miou.add({'miou': miou})

                ious.update(miou.item(), B)



            miou = (avg_meter_miou.pop('miou'))
            if miou > max_miou:
                model_save_path = os.path.join(checkpoint_dir, '%s_best.pth'%(args.session_name))
                torch.save(model.module.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s'%model_save_path)

            miou_list.append(miou)
            max_miou = max(miou_list)

            val_log = 'Epoch: {}, Miou: {}, maxMiou: {}'.format(epoch, miou, max_miou)
            # print(val_log)
            logger.info(val_log)

        args.train_plotter.add_data('val/miou', ious.avg, epoch)

        model.train()
    logger.info('best val Miou {} at epoch: {}'.format(max_miou, best_epoch))











