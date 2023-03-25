import torch
import torch.nn as nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, predict, target):
        predict = predict.contiguous().view(predict.shape[0], -1)  # [bs, c*w*h]
        target = target.contiguous().view(target.shape[0], -1)  # [bs, c*w*h]

        num = (predict * target).sum(1)  # [bs,]
        den = predict.sum(1) + target.sum(1)  # [bs,]

        loss = 1 - (2 * num + 1e-10) / (den + 1e-10)  # [bs,]

        return loss.mean()  # [1]


class LogExpDiceLoss(nn.Module):
    def __init__(self):
        super(LogExpDiceLoss, self).__init__()

    def forward(self, predict, target):
        dice = BinaryDiceLoss()
        total_loss = 0

        for i in range(target.shape[1]):
            dice_loss = dice(predict[:, i], target[:, i])
            total_loss += torch.pow(-torch.log(1 - dice_loss), .3)

        return total_loss / target.shape[1]


class ExpWCELoss(nn.Module):
    def __init__(self):
        super(ExpWCELoss, self).__init__()

    def weight_function(self, target):
        # weight_l = (# {voxels}/ # {class_l}))^{1/2}
        mask = torch.argmax(target, dim=1)
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        weights = []
        for i in range(mask.max() + 1):
            voxels_i = [mask == i][0].sum().cpu().numpy()
            w_i = np.sqrt(voxels_sum / voxels_i).astype(np.float32)
            weights.append(w_i)
        weights = torch.from_numpy(np.array(weights)).to(device)
        return weights

    def forward(self, predict, target):
        ce_loss = torch.mean(-target * torch.log(predict + 1e-10), dim=(0, 2, 3))  # (out_ch, )
        weights = self.weight_function(target)
        loss = ce_loss * weights

        return loss.mean()



class SegmentationLoss(nn.Module):
    """
    segmentation loss = 0.8 * Log-Exp dice loss + 0.2 * exp weighted cross entropy loss
    """

    def __init__(self):
        super(SegmentationLoss, self).__init__()

        self.dice_loss = LogExpDiceLoss()
        self.ce_loss = ExpWCELoss()


    def forward(self, predict, target):
        dice_loss = self.dice_loss(predict, target)
        ce_loss = self.ce_loss(predict, target)
        loss = 0.2 * ce_loss + 0.8 * dice_loss

        return loss


class ModuleLoss(nn.Module):
    def __init__(self):
        super(ModuleLoss, self).__init__()

        self.seg_loss = SegmentationLoss()
        self.fn_loss = nn.MSELoss(reduction='sum')

    def forward(self, seg, label, y_true, y_pred, w):
        # y = r in SRNN, y = p in SCN
        loss_seg = self.seg_loss(seg, label)    # seg_loss
        loss_fn = torch.sqrt(self.fn_loss(y_pred, y_true))  # fn_loss

        loss = loss_seg + w * loss_fn  # total loss

        return loss_seg, loss_fn, loss


class SRSCNLoss(nn.Module):
    def __init__(self):
        super(SRSCNLoss, self).__init__()

        self.seg_loss = SegmentationLoss()
        self.fn_loss = nn.MSELoss(reduction='sum')

    def forward(self, seg, label, p_true, p_pred, r_true, r_pred, w_sc, w_sr):
        loss_seg = self.seg_loss(seg, label)    # seg_loss
        loss_sc = torch.sqrt(self.fn_loss(p_pred, p_true))  # sc_loss
        loss_sr = torch.sqrt(self.fn_loss(r_pred, r_true))  # sr_loss

        loss = loss_seg + w_sc * loss_sc + w_sr * loss_sr  # total loss

        return loss_seg, loss_sc, loss_sr, loss
