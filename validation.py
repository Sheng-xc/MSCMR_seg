import torch
from torchvision.utils import make_grid
from utils import LargestConnectedComponents, Normalization
from criterion.metrics import Criterion, DiceMeter, Logger2d


@torch.no_grad()
def Validation2d(args, epoch, model, valid_image, valid_loader, writer, log_name, tensorboardImage):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = Logger2d()
    criterion = Criterion()
    normalize = Normalization()
    keepLCC = LargestConnectedComponents()
    valid = DiceMeter()

    test_DE = torch.FloatTensor(1, 1, args.dim, args.dim).to(device)
    cardiac_gd = torch.FloatTensor(args.dim, args.dim)

    # write to tensorboard
    if tensorboardImage == True:
        pic = torch.FloatTensor(3 * int(len(valid_image)), 1, args.dim, args.dim).to(device)

    for iter in range(int(len(valid_image))):
        img_DE, cardiac_label, imgidx = next(valid_loader)

        test_DE.copy_(img_DE)
        cardiac_gd.copy_(cardiac_label[0, 0, ...])


        _, res = model(test_DE)  # model output


        seg_DE = torch.argmax(res, dim=1).squeeze(0)  # one-hot (b,c,w,h) ->  (w,h)
        seg_DE = keepLCC(seg_DE.cpu())                # keep the largest connected area

        myo, lv, rv = criterion(seg_DE, cardiac_gd)  # calc_dice
        valid.update(myo, lv, rv)                    # dice meter

        if tensorboardImage == True:
            pic[3 * iter:3 * iter + 3, ...].copy_(torch.cat([
                torch.from_numpy(normalize(test_DE.detach().cpu().numpy(), 'Truncate')),
                normalize(cardiac_gd, 'Max_Min').unsqueeze(0).unsqueeze(0),
                normalize(seg_DE, 'Max_Min').unsqueeze(0).unsqueeze(0)], dim=0))

    if tensorboardImage == True:
        writer.add_image('pic', make_grid(pic, nrow=9, padding=2), epoch)

    dice = {'myo': valid.myo['avg'], 'lv': valid.lv['avg'], 'rv': valid.rv['avg']}

    # write to log
    logger(epoch + 1, args.end_epoch, log_name, dice)

    # write to tensorboard
    writer.add_scalar('valid MYO 2d', dice['myo'], epoch)
    writer.add_scalar('valid LV 2d', dice['lv'], epoch)
    writer.add_scalar('valid RV 2d', dice['rv'], epoch)

    avg_dice = (dice['myo'] + dice['lv'] + dice['rv']) / 3

    return avg_dice
