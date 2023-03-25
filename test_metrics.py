import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from medpy import metric
from utils.config import config


# extract ROI
def get_ROI(img, index):
    for key, value in index.items():
        img = np.where(img == key, value, img)
    return img


# gd_dice_
def calculate(seg_roi, gd_roi, index):
    labels = np.array([0, 200, 500, 600])
    if index == 200:  # myo
        trans_roi = np.array([0, 1, 0, 0])
    if index == 500:  # lv
        trans_roi = np.array([0, 0, 1, 0])
    if index == 600:  # rv
        trans_roi = np.array([0, 0, 0, 1])

    gd_ = sitk.ReadImage(gd_roi)
    spacing = gd_.GetSpacing()
    gd = sitk.GetArrayFromImage(gd_)
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_roi))

    seg_roi = get_ROI(seg, dict(zip(labels, trans_roi)))
    gd_roi = get_ROI(gd, dict(zip(labels, trans_roi)))

    # dice = 2 * TP / ((TP + FP) + (TP + FN))
    try:
        dice = metric.binary.dc(seg_roi, gd_roi)
    except:
        dice = 0

    # hd
    for i in range(gd.shape[0]):
        hds = []
        try:
            hds.append(metric.hd(seg_roi[i,...], gd_roi[i, ...], voxelspacing=(spacing[0], spacing[1])))
        except:
            pass
    # final hd as mean over all slices of a subject
    try:
        hd = sum(hds)/len(hds)
    except:
        hd = 0

    # assd
    for i in range(gd.shape[0]):
        assds = []
        try:
            assds.append(metric.binary.assd(seg_roi[i,...], gd_roi[i,...], voxelspacing=(spacing[0], spacing[1])))
        except:
           pass
    # final assd as mean over all slices of a subject
    try:
        assd = sum(assds)/len(assds)
    except:
        assd = 0

    return dice, hd, assd


def main(args):
    result = []
    res_path = Path(args.test_path) / 'test_15'

    for file in res_path.glob('*.nii.gz'):

        subject = '_'.join(file.name.split('.')[0].split('_')[:-1])
        gd = Path(args.path) / 'mscmr_manual' / f'{subject}_manual.nii.gz'

        try:
            dice_m, hd_m, assd_m = calculate(file, gd, 200)  # myo
        except:
            dice_m, hd_m, assd_m = 'Null', 'Null', 'Null'

        try:
            dice_l, hd_l, assd_l = calculate(file, gd, 500)  # lv
        except:
            dice_l, hd_l, assd_l = 'Null', 'Null', 'Null'

        try:
            dice_r, hd_r, assd_r = calculate(file, gd, 600)  # rv
        except:
            dice_r, hd_r, assd_r = 'Null', 'Null', 'Null'

        result.append((subject, dice_m, hd_m, assd_m, dice_l, hd_l, assd_l, dice_r, hd_r, assd_r))

    result_df = pd.DataFrame(result, columns=['test_sub', 'myo_dice', 'myo_hd', 'myo_assd', 'lv_dice',
                                              'lv_hd', 'lv_assd', 'rv_dice', 'rv_hd', 'rv_assd'])
    result_df.to_csv(f'{args.test_path}_result.csv', index=False)
    print("Done!")


if __name__ == '__main__':
    args = config()
    main(args)
