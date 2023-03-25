import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from network import UNet
from utils import config, LargestConnectedComponents, Normalization, ImageTransform, ResultTransform


def predict(args, model_path, epoch):
    # model definition
    model_type = Path(args.load_path).name
    model = UNet(in_ch=1, out_ch=4)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    if not Path.exists(Path(f'test/{model_type}/test_' + str(epoch))):
        Path(f'test/{model_type}/test_' + str(epoch)).mkdir(parents=True, exist_ok=True)

    # 读取测试数据列表
    test_img = pd.read_table(args.path + '/test.txt', sep='\t', header=None)

    normalize = Normalization()
    keepLCC = LargestConnectedComponents()  # 获取最大联通块类
    image_transform = ImageTransform(args.dim, 'Test')  # 图像增强
    result_transform = ResultTransform(ToOriginal=True)  # 结果转换

    for i in range(int(len(test_img))):

        # raw image
        img_path = Path(args.path) / test_img.iloc[i, 0]
        dim_x, dim_y, dim_z = test_img.iloc[i, 1:]
        result = torch.zeros([dim_z, dim_x, dim_y])

        DE_raw = nib.load(img_path)
        img_affine = DE_raw.affine

        # preprocessing
        DE_img = normalize(DE_raw.get_fdata(), 'Truncate').astype(np.float32)
        img_DE = image_transform(DE_img)

        # segmentation results
        test_DE = torch.FloatTensor(1, 1, args.dim, args.dim)
        seg_DE = torch.FloatTensor(dim_z, args.dim, args.dim)

        for j in range(dim_z):
            img_DE_slice = normalize(img_DE[j:j + 1, ...], 'Zero_Mean_Unit_Std')
            test_DE.copy_(img_DE_slice.unsqueeze(0))  # store 2D slices, (1,1,w,h)
            _, res_DE = model(test_DE)  # one-hot segmentation result
            seg_DE[j:j + 1, :, :].copy_(torch.argmax(res_DE, dim=1))  # one-hot -> encoding

        # post process
        seg_DE = result_transform(keepLCC(seg_DE))

        result[:, dim_x // 2 - args.dim // 2:dim_x // 2 + args.dim // 2,
        dim_y // 2 - args.dim // 2:dim_y // 2 + args.dim // 2].copy_(seg_DE)

        result = result.numpy().transpose(1, 2, 0)
        seg_map = nib.Nifti1Image(result, img_affine)
        nib.save(seg_map,
                 f'test/{model_type}/test_' + str(epoch) + '/' +
                 test_img.iloc[i, 0].split('/')[-1].split('.')[0] + '_result.nii.gz')
        print(test_img.iloc[i, 0] + "_Successfully saved!")


def predict_multiple(args):
    if not Path.exists(Path('test')):
        os.makedirs(Path('test'))

    load_path = Path(args.load_path)
    for model_path in load_path.glob('*.pth'):
        dice = float(str(model_path.name).split('[')[0])
        epoch = int((str(model_path.name).split('[')[1]).split(']')[0])
        if args.predict_mode == 'single':
            if dice == args.threshold:
                print('--- Start predicting epoch ' + str(epoch) + ' ---')
                predict(args, model_path, epoch)
                print('--- Test done for epoch ' + str(epoch) + ' ---')
        if args.predict_mode == 'multiple':
            if dice > args.threshold:
                print('--- Start predicting epoch ' + str(epoch) + ' ---')
                predict(args, model_path, epoch)
                print('--- Test done for epoch ' + str(epoch) + ' ---')


if __name__ == '__main__':
    args = config()
    predict_multiple(args)
