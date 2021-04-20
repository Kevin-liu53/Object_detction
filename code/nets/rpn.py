import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils import loc2bbox


class ProposalCreator():
    def __init__(self, mode, nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=600,
                 n_test_pre_nms=3000,
                 n_test_post_nms=300,
                 min_size=16):
        self.mode = mode
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        """

        :param loc:        预测的偏移量 dx,dy,dw,dh (9*h*w,4)
        :param score:       预测的正样本的概率（9*h*w,2)
        :param anchor:     anchorbox坐标（9*h*w,4)
        :param img_size:
        :param scale:
        :return:
        """
        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        anchor = torch.from_numpy(anchor)
        if loc.is_cuda:
            anchor = anchor.cuda()
        # -----------------------------------#
        #   将RPN网络预测结果转化成建议框（x1,y1,x2,y2)  anchor + loc = proposal  因为loc放的是偏移量
        # -----------------------------------#
        roi = loc2bbox(anchor, loc)

        # -----------------------------------#
        #   防止建议框超出图像边缘
        # -----------------------------------#
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])

        # -----------------------------------#
        #   建议框的宽高的最小值不可以小于16
        # -----------------------------------#
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # -----------------------------------#
        #   根据得分进行排序，取出建议框
        # -----------------------------------#
        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # -----------------------------------#
        #   对建议框进行非极大抑制
        # -----------------------------------#
        keep = nms(roi, score, self.nms_thresh)
        keep = keep[:n_post_nms]       # 只取了300个proposal
        roi = roi[keep]
        return roi


class RegionProposalNetwork(nn.Module):

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            mode="training",
    ):
        super(RegionProposalNetwork, self).__init__()
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(mode)  #每一个点产生9行4列的位置尺寸
        # -----------------------------------------#
        #   生成基础先验框，shape为[9, 4]
        # -----------------------------------------#
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        n_anchor = self.anchor_base.shape[0]

        # -----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        # -----------------------------------------#
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # -----------------------------------------#
        #   分类预测先验框内部是否包含物体
        # -----------------------------------------#
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # -----------------------------------------#
        #   回归预测对先验框进行调整
        # -----------------------------------------#
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        # --------------------------------------#
        #   对FPN的网络部分进行权值初始化
        # --------------------------------------#
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """

        :param x:    特征图
        :param img_size: 源图像尺寸
        :param scale:
        :return:    rpn_locs:预测的位置偏移量
                    rpn_scores:预测的每一个框的正负样本的概率
                    rois:proposal框的坐标(x1,y1,x2,y2)
                    roi_indices: 每一张图片一个索引
                    anchor：生成的anchor坐标(x1,y1,x2,y2)to(xi,yi)  固定坐标
        """
        n, _, h, w = x.shape
        # -----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        # -----------------------------------------#
        x = F.relu(self.conv1(x))
        # -----------------------------------------#
        #   回归预测对先验框位置进行调整
        # -----------------------------------------#
        rpn_locs = self.loc(x)    #(n,9*4,h,w)->(n,h,w,9*4)->(n,9*h*w,4)  也就是每一行表示一个proposal，每一行的元素表示坐标
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # -----------------------------------------#
        #   分类预测先验框内部是否包含物体
        # -----------------------------------------#
        rpn_scores = self.score(x) #（n,9*2,h,w)->(n,h,w,9*2)->(n,9*h*w,2) 每一行表示一个proposal,每个行的元素表示正负样本的分数
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)

        # --------------------------------------------------------------------------------------#
        #   进行softmax概率计算，每个先验框只有两个判别结果
        #   内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        # --------------------------------------------------------------------------------------#
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)  #对正负样本进行softmax回归，得到正负样本的分类概率
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous() #指定内部不包含物体维第1维
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        #将其展开成batch_size*(9*h*w)的形式，即每一行表示一张图像，每一行的元素表示特征图从第一个元素到最后一个元素是正样本的概率

        # ------------------------------------------------------------------------------------------------#
        #   生成先验框(在原始图像中的坐标)，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
        # ------------------------------------------------------------------------------------------------#
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)

        rois = list()
        roi_indices = list()
        for i in range(n):
            #(300,4)
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)#特征图上所有选出来的proposal （x1,y1,x2,y2)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = torch.cat(rois, dim=0) #竖着合并  （n*300,4)
        roi_indices = torch.cat(roi_indices, dim=0) #(n*300,4)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()