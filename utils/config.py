import argparse


def config(name='SRSCN'):
    parser = argparse.ArgumentParser(description=name)

    # data path
    parser.add_argument('--path', type=str, default='/home/shengxicheng/Data/MSCMR', help="data path")

    # predict & test
    parser.add_argument('--load_path', type=str, default='checkpoints/unet', help="load path")
    parser.add_argument('--predict_mode', type=str, default='single', help="predict mode: single or multiple")
    parser.add_argument('--test_path', type=str, default='test/unet', help="test path")

    # parameters
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--dim', type=int, default=240, help="dimension of 2D image")
    parser.add_argument('--lr', type=float, default=1e-4, help="starting learning rate")
    parser.add_argument('--weight_sc', type=float, default=5e-4, help="weight for spatial constraint loss")
    parser.add_argument('--weight_sr', type=float, default=5e-4, help="weight for shape restriction loss")

    # settings
    parser.add_argument('--threshold', type=float, default=0.6477869023688831, help="the minimum dice to predict model")

    parser.add_argument('--start_epoch', type=int, default=0, help="flag to indicate the start epoch")
    parser.add_argument('--end_epoch', type=int, default=30, help="flag to indicate the final epoch")

    args = parser.parse_args()

    return args
