import torch
import random
import numpy as np
import torch.nn as nn
from torchvision import transforms
from skimage.transform import resize


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Normalization(object):
    """
    normalization mode can be chosen from 'Max_Min', 'Zero_Mean_Unit_Std', and 'Truncate'(truncate before normalization)
    """

    def __call__(self, image, mode):
        # shape (image) = 3

        if mode == 'Max_Min':
            eps = 1e-8
            mn = image.min()
            mx = image.max()
            image_normalized = (image - mn) / (mx - mn + eps)

        if mode == 'Zero_Mean_Unit_Std':
            eps = 1e-8
            mean = image.mean()
            std = image.std()
            image_normalized = (image - mean) / (std + eps)

        if mode == 'Truncate':
            # truncate

            Hist, _ = np.histogram(image, bins=int(image.max()))  # np.histogram returns: hist array, boundary array

            idexs = np.argwhere(Hist >= 20)
            idex_min = np.float32(0)
            idex_max = np.float32(idexs[-1, 0])

            image[np.where(image <= idex_min)] = idex_min
            image[np.where(image >= idex_max)] = idex_max

            image_normalized = image

        return image_normalized


class RandomSizedCrop(object):
    """
    Random crop and resize image to  (1, crop_size, crop_size)
    """

    def __init__(self, dim):
        self.crop_size = dim

    def __call__(self, image):
        # RandomCrop
        scaler = np.random.uniform(0.9, 1.1)
        scale_size = int(self.crop_size * scaler)
        h_off = random.randint(0, image.shape[1] - 0 - scale_size)
        w_off = random.randint(0, image.shape[2] - 0 - scale_size)
        image = image[:, h_off:h_off + scale_size, w_off:w_off + scale_size]

        # Resize
        image = image.numpy()

        DE_slice = image[:1, ...]
        label_slice = image[1:, ...]

        output_shape = (1, self.crop_size, self.crop_size)

        DE_resized = resize(DE_slice, output_shape, order=1, mode='constant', preserve_range=True)
        label_resized = resize(label_slice, output_shape, order=0, mode='edge', preserve_range=True)

        image = np.concatenate([DE_resized, label_resized], axis=0)
        image = torch.from_numpy(image).float()
        return image


class ToTensor(object):
    """
    convert imported image(W,H,C) to tensor (C,W,H)
    """

    def __call__(self, image):
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = torch.from_numpy(image).float()
        return image


# image transform
class ImageTransform(object):
    def __init__(self, dim, stage):
        self.dim = dim
        self.stage = stage

    def __call__(self, image):

        if self.stage == 'Train':
            transform = transforms.Compose([
                ToTensor(),
                transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), shear=0.1),
                RandomSizedCrop(self.dim)
            ])

        if self.stage == 'Valid' or self.stage == 'Test':
            transform = transforms.Compose([
                ToTensor(),
                transforms.CenterCrop(self.dim)
            ])

        return transform(image)


class LabelTransform(object):
    """
    convert label encodings, and return gt for cardiac segmentation.
    """

    def __init__(self, stage):
        self.stage = stage

    def __call__(self, label):
        label = label.numpy()
        cardiac_gd = self.label_transform(label)

        if self.stage == 'Train':
            cardiac_gd = self.convert_onehot(cardiac_gd, 4)  # bg myo lv rv

        return cardiac_gd

    @staticmethod
    def convert_onehot(label, num_class):
        """
        :param label: (1,w,h)
        :param num_class:
        :return: (c,w,h)
        """
        label = label.long()
        label_onehot = torch.zeros((num_class, label.shape[1], label.shape[2]))
        label_onehot.scatter_(0, label, 1).float()
        return label_onehot

    # label transform
    @staticmethod
    def label_transform(label):
        # 200 - myo, 500 - lv, 600 - rv, 0 - bg
        label = np.where(label == 200, 1, label)
        label = np.where(label == 500, 2, label)
        label = np.where(label == 600, 3, label)

        label = torch.from_numpy(label).float()
        return label


# result transform
class ResultTransform(object):
    def __init__(self, ToOriginal=False):
        self.flag = ToOriginal

    def __call__(self, seg_cardiac):
        seg_cardiac = seg_cardiac.numpy()

        if self.flag == True:
            seg_cardiac = np.where(seg_cardiac == 1, 200, seg_cardiac)  # 1 - myo - 200
            seg_cardiac = np.where(seg_cardiac == 2, 500, seg_cardiac)  # 2 - lv  - 500
            seg_cardiac = np.where(seg_cardiac == 3, 600, seg_cardiac)  # 3 - rv  - 500

        seg_cardiac = torch.from_numpy(seg_cardiac)

        return seg_cardiac


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import nibabel as nib
    sample_slice = nib.load('../../Data/MSCMR/mscmr_image/subject10_DE.nii.gz').get_fdata()[:, :, 8].transpose((1, 0))
    sample_gt = nib.load('../../Data/MSCMR/mscmr_manual/subject10_DE_manual.nii.gz').get_fdata()[:, :, 8].transpose((1, 0))

    normalize = Normalization()

    plt.figure(1)

    # img
    plt.subplot(2, 4, 1)
    plt.imshow(sample_slice, cmap='gray')

    plt.subplot(2, 4, 2)
    plt.imshow(normalize(sample_slice, 'Max_Min'), cmap='gray')

    plt.subplot(2, 4, 3)
    plt.imshow(normalize(sample_slice, 'Zero_Mean_Unit_Std'), cmap='gray')

    plt.subplot(2, 4, 4)
    plt.imshow(normalize(sample_slice, 'Truncate'), cmap='gray')

    # label
    plt.subplot(2, 4, 5)
    plt.imshow(sample_gt, cmap='gray')

    plt.subplot(2, 4, 6)
    plt.imshow(normalize(sample_gt, 'Max_Min'), cmap='gray')

    plt.subplot(2, 4, 7)
    plt.imshow(normalize(sample_gt, 'Zero_Mean_Unit_Std'), cmap='gray')

    plt.subplot(2, 4, 8)
    plt.imshow(normalize(sample_gt, 'Truncate'), cmap='gray')

    plt.show()




