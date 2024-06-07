import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import build_transformer, build_positional_encoding, build_fusion_block, build_generator
from ops.modules import MSDeformAttn
from torch.nn.init import normal_
from torch.nn.functional import interpolate
from torchvision.models import resnet18 
import math 

class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class Self_Attn(nn.Module):
    """ Self attention Layer """
    def __init__(self, in_dim, key_in_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=key_in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, v):
        """
            inputs:
                x: input feature maps (B x C x H x W)
            returns:
                out: self attention value + input feature
                attention: B x N x N (N is Width * Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key) / math.sqrt(self.channel_in//8)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        return out, attention


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()

        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Conv2d(n, k, kernel_size=1, stride=1, padding=0)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SimpleFPN(nn.Module):
    def __init__(self, channel=256, layers=3):
        super().__init__()

        assert layers == 3  # only designed for 3 layers
        self.up1 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        self.up2 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        self.up3 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x1 = self.up1(x[-1])
        x1 = x1 + x[-2]

        x2 = self.up2(x1)
        x2 = x2 + x[-3]

        y = self.up3(x2)
        return y


class AVSegHeadUFE(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 query_num,
                 transformer,
                 query_generator,
                 embed_dim=256,
                 valid_indices=[1, 2, 3],
                 scale_factor=4,
                 positional_encoding=None,
                 use_learnable_queries=True,
                 fusion_block=None) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = query_num
        self.valid_indices = valid_indices
        self.num_feats = len(valid_indices)
        self.scale_factor = scale_factor
        self.use_learnable_queries = use_learnable_queries
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feats, embed_dim))
        self.learnable_query = nn.Embedding(query_num, embed_dim)

        self.query_generator = build_generator(**query_generator)

        self.transformer = build_transformer(**transformer)
        if positional_encoding is not None:
            self.positional_encoding = build_positional_encoding(
                **positional_encoding)
        else:
            self.positional_encoding = None

        in_proj = []
        for c in in_channels:
            in_proj.append(
                nn.Sequential(
                    nn.Conv2d(c, embed_dim, kernel_size=1),
                    nn.GroupNorm(32, embed_dim)
                )
            )
        self.in_proj = nn.ModuleList(in_proj)
        self.mlp = MLP(query_num, 2048, embed_dim, 3)

        if fusion_block is not None:
            self.fusion_block = build_fusion_block(**fusion_block)

        self.lateral_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, embed_dim)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, embed_dim),
            nn.ReLU(True)
        )

        self.fpn = SimpleFPN()
        self.attn_fc = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=scale_factor, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.fc = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=scale_factor, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )

        self._reset_parameters()

        # DEBUG
        # import ipdb; ipdb.set_trace()
        channel = 256 
        self.conv4_flow = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 64)
        self.conv3_flow = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 64)
        self.conv2_flow = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 64)
        self.conv1_flow = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 64)
        self.flownet = resnet18(pretrained=True)
        self.flownet.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.flownet.avgpool = nn.Identity()
        self.flownet.fc = nn.Identity()
        # self.load_pretrained_flownet()
        # self.flowatt = Self_Attn(512, 512)
        # self.flowatt = Self_Attn(256, 256)
        # self.flow_linear = nn.Linear(512, 256)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def load_pretrained_flownet(self, checkpoint_path="/remote-home/share/yikunliu/AVSegFormer-master/pretrained_flownet.pth"):
        checkpoint = torch.load(checkpoint_path)
        self.flownet.load_state_dict(checkpoint)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def reform_output_squences(self, memory, spatial_shapes, level_start_index, dim=1):
        split_size_or_sections = [None] * self.num_feats
        for i in range(self.num_feats):
            if i < self.num_feats - 1:
                split_size_or_sections[i] = level_start_index[i +
                                                              1] - level_start_index[i]
            else:
                split_size_or_sections[i] = memory.shape[dim] - \
                    level_start_index[i]
        y = torch.split(memory, split_size_or_sections, dim=dim)
        return y

    def forward(self, feats, audio_feat, flow, normalize=True):
        flow = self.flownet.conv1(flow)
        flow = self.flownet.bn1(flow)
        flow = self.flownet.relu(flow)
        flow = self.flownet.maxpool(flow)
        flow1 = self.flownet.layer1(flow)
        conv1_flow_feat = self.conv1_flow(flow1)
        conv2_flow_feat = self.conv2_flow(flow1)
        conv3_flow_feat = self.conv3_flow(flow1)
        conv4_flow_feat = self.conv4_flow(flow1)
        # flow = self.flownet(flow).view(-1, 512, 7, 7)
        # flow = self.flow_linear(flow.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view(-1, 256, 7, 7)
        feat14 = self.in_proj[0](feats[0])
        # feat14_attention, _ = self.flowatt(feat14, flow)
        # feat14 = feat14 + feat14_attention
        srcs = [self.in_proj[i](feats[i]) for i in self.valid_indices]
        conv1_flow_feat = nn.functional.interpolate(conv1_flow_feat, size=feat14.size()[-2:], mode='bilinear')
        conv2_flow_feat = nn.functional.interpolate(conv2_flow_feat, size=srcs[0].size()[-2:], mode='bilinear')
        conv3_flow_feat = nn.functional.interpolate(conv3_flow_feat, size=srcs[1].size()[-2:], mode='bilinear')
        conv4_flow_feat = nn.functional.interpolate(conv4_flow_feat, size=srcs[2].size()[-2:], mode='bilinear')
        conv1_flow_feat =  nn.functional.normalize(conv1_flow_feat, dim=1)
        conv2_flow_feat =  nn.functional.normalize(conv2_flow_feat, dim=1)
        conv3_flow_feat =  nn.functional.normalize(conv3_flow_feat, dim=1)
        conv4_flow_feat =  nn.functional.normalize(conv4_flow_feat, dim=1)
        feat14 = nn.functional.normalize(feat14, dim=1)
        srcs[0] = nn.functional.normalize(srcs[0], dim=1)
        srcs[1] = nn.functional.normalize(srcs[1], dim=1)
        srcs[2] = nn.functional.normalize(srcs[2], dim=1)
        # srcs = [self.flowatt(src, flow)[0] + src for src in srcs]
        feat14 = feat14 + conv1_flow_feat
        srcs[0] = srcs[0] + conv2_flow_feat
        srcs[1] = srcs[1] + conv3_flow_feat
        srcs[2] = srcs[2] + conv4_flow_feat
        feat14 = nn.functional.normalize(feat14, dim=1)
        srcs[0] = nn.functional.normalize(srcs[0], dim=1)
        srcs[1] = nn.functional.normalize(srcs[1], dim=1)
        srcs[2] = nn.functional.normalize(srcs[2], dim=1)
        masks = [torch.zeros((x.size(0), x.size(2), x.size(
            3)), device=x.device, dtype=torch.bool) for x in srcs]
        pos_embeds = []
        for m in masks:
            pos_embeds.append(self.positional_encoding(m))
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # prepare queries
        bs = audio_feat.shape[0]
        query = self.query_generator(audio_feat)
        if self.use_learnable_queries:
            query = query + \
                self.learnable_query.weight[None, :, :].repeat(bs, 1, 1)

        memory, outputs = self.transformer(query, src_flatten, spatial_shapes,
                                           level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # generate mask feature
        mask_feats = []
        for i, z in enumerate(self.reform_output_squences(memory, spatial_shapes, level_start_index, 1)):
            mask_feats.append(z.transpose(1, 2).view(
                bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        cur_fpn = self.lateral_conv(feat14)
        mask_feature = mask_feats[0]
        mask_feature = cur_fpn + \
            F.interpolate(
                mask_feature, size=cur_fpn.shape[-2:], mode='bilinear', align_corners=False)
        mask_feature = self.out_conv(mask_feature)
        if hasattr(self, 'fusion_block'):
            mask_feature = self.fusion_block(mask_feature, audio_feat)

        # predict output mask
        pred_feature = torch.einsum(
            'bqc,bchw->bqhw', outputs[-1], mask_feature)
        pred_feature = self.mlp(pred_feature)
        pred_mask = mask_feature + pred_feature
        pred_mask = self.fc(pred_mask)

        return pred_mask, mask_feature

    # def forward_prediction_head(self, output, mask_embed, spatial_shapes, level_start_index):
    #     masks = torch.einsum('bqc,bqn->bcn', output, mask_embed)
    #     splitted_masks = self.reform_output_squences(
    #         masks, spatial_shapes, level_start_index, 2)

    #     bs = output.shape[0]
    #     reforms = []
    #     for i, embed in enumerate(splitted_masks):
    #         embed = embed.view(
    #             bs, -1, spatial_shapes[i][0], spatial_shapes[i][1])
    #         reforms.append(embed)

    #     attn_mask = self.fpn(reforms)
    #     attn_mask = self.attn_fc(attn_mask)
    #     return attn_mask
