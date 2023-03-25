from pathlib import Path
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from utils.tools import Normalization, ImageTransform, LabelTransform


class DEDataLoader(Dataset):

    def __init__(self, path, file_name, dim, max_iters=None, stage='Train'):

        self.path = Path(path)
        self.crop_size = dim
        self.stage = stage

        # item.strip().split()给出了file里每行的[img_path, gt_path, slice_idx]
        self.Img = [item.strip().split() for item in open(self.path/file_name)]  # e.g. file_name = train.txt

        if max_iters != None:
            self.Img = self.Img * int(np.ceil(float(max_iters) / len(self.Img)))

        self.files = []

        for item in self.Img:
            img_path, gt_path, imgidx = item

            img_file = self.path/img_path
            label_file = self.path/gt_path

            self.files.append({
                "DE": img_file,
                "label": label_file,
                "index": int(imgidx)
            })

        self.normalize = Normalization()
        self.image_transform = ImageTransform(self.crop_size, self.stage)
        self.label_transform = LabelTransform(self.stage)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file_path = self.files[index]

        # get raw data
        DE_raw = nib.load(file_path["DE"])
        gd_raw = nib.load(file_path["label"])
        imgidx = file_path["index"]

        # get data [x,y,z] & normalize
        DE_img = self.normalize(DE_raw.get_fdata(), 'Truncate')
        gd_img = gd_raw.get_fdata()

        # cut slice [x,y,1] -> [2,x,y]
        DE_slice = DE_img[:, :, imgidx:imgidx + 1].astype(np.float32)
        label_slice = gd_img[:, :, imgidx:imgidx + 1].astype(np.float32)
        image = np.concatenate([DE_slice, label_slice], axis=2)
        img_DE, label = torch.chunk(self.image_transform(image), chunks=2, dim=0)

        # image tranform [1,W,H]
        img_DE = self.normalize(img_DE, 'Zero_Mean_Unit_Std')

        # label transform [class,W,H]
        label_cardiac = self.label_transform(label)

        return img_DE, label_cardiac, imgidx/DE_img.shape[2]
