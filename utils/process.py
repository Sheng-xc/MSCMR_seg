import os
import torch
import numpy as np
import nibabel as nib
from skimage import measure


class LargestConnectedComponents(object):
    def __call__(self, mask):

        mask = mask.numpy()

        # keep a heart connectivity
        heart_slice = np.where((mask > 0), 1, 0)
        out_heart = np.zeros(heart_slice.shape, dtype=np.uint8)
        for struc_id in [1]:
            binary_img = (heart_slice == struc_id)
            blobs = measure.label(binary_img, connectivity=1)  # connected blobs(1-neighbor)
            props = measure.regionprops(blobs)  # properties of the blobs
            if not props:
                continue  # skip if no connected blobs
            area = [ele.area for ele in props]  # area of the blobs
            largest_blob_ind = np.argmax(area)
            largest_blob_label = props[largest_blob_ind].label
            out_heart[blobs == largest_blob_label] = struc_id   # largest connected area == "struc_id"

        # keep MYO/LV connectivity
        out_img = np.zeros(mask.shape, dtype=np.uint8)
        for struc_id in [1, 2]:
            binary_img = mask == struc_id
            blobs = measure.label(binary_img, connectivity=1)
            props = measure.regionprops(blobs)
            if not props:
                continue
            area = [ele.area for ele in props]
            largest_blob_ind = np.argmax(area)
            largest_blob_label = props[largest_blob_ind].label
            out_img[blobs == largest_blob_label] = struc_id

        final_img = out_heart * (out_img + np.where(mask == 3, 3, 0))  # and add RV
        final_img = torch.from_numpy(final_img).float()

        return final_img