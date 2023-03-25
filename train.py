from itertools import cycle
from pathlib import Path
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from criterion import SegmentationLoss, ModuleLoss, SRSCNLoss
from utils import weights_init, DEDataLoader
from network import UNet, ConSC, ConSR
from validation import Validation2d


def SegNetTrain(args, model_type):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet(in_ch=1, out_ch=4).to(device)
    srnet = ConSR(4).to(device)
    srnet.load_state_dict(torch.load('checkpoints/sc.pth'))
    for param in srnet.parameters():
        param.requires_grad = False

    if model_type == 'unet':
        net_loss = SegmentationLoss().to(device)

    elif model_type == 'scn':
        scnet = ConSC().to(device)
        scnet.apply(weights_init)
        net_loss = ModuleLoss().to(device)

    elif model_type == 'srnn':
        net_loss = ModuleLoss().to(device)

    elif model_type == 'srscn':
        scnet = ConSC().to(device)
        scnet.apply(weights_init)
        net_loss = SRSCNLoss().to(device)

    else:
        raise ValueError(f"Unknown model typr: {model_type}!\nPlease choose from 'unet','scn','srnn' and 'srscn'")

    model.apply(weights_init)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6, last_epoch=-1)

    if not Path.exists(Path(f'checkpoints/{model_type}')):
        Path(f'checkpoints/{model_type}').mkdir(parents=True, exist_ok=True)

    if not Path.exists(Path('logs')):
        Path('logs').mkdir()

    Train_Image = DEDataLoader(path=args.path, file_name='train.txt', dim=args.dim,
                               max_iters = 54 * args.batch_size, stage='Train')
    Train_loader = cycle(
        DataLoader(Train_Image, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True))

    Valid_Image = DEDataLoader(path=args.path, file_name='valid.txt', dim=args.dim, max_iters=None, stage='Valid')
    Valid_loader = cycle(DataLoader(Valid_Image, batch_size=1, shuffle=False, num_workers=0, drop_last=False))

    # write to tensor board
    writer = SummaryWriter('logs/board', comment=model_type)

    for epoch in range(args.start_epoch, args.end_epoch):

        # Train
        model.train()

        train_DE = torch.FloatTensor(args.batch_size, 1, args.dim, args.dim).cuda()
        cardiac_gd = torch.FloatTensor(args.batch_size, 4, args.dim, args.dim).cuda()
        imgidx_gd = torch.FloatTensor(args.batch_size, 1).cuda()

        IterCount = int(len(Train_Image) / args.batch_size)  # of Iterations for each epoch

        for iteration in range(IterCount):
            img_DE, label_cardiac, imgidx = next(Train_loader)

            train_DE.copy_(img_DE)
            cardiac_gd.copy_(label_cardiac)
            imgidx_gd.copy_(imgidx.unsqueeze(1))

            encoding, seg_DE = model(train_DE)

            if model_type == 'unet':
                loss = net_loss(seg_DE, cardiac_gd)

            elif model_type == 'scn':
                code = scnet(encoding)
                loss_seg, loss_sc, loss = net_loss(seg_DE, cardiac_gd, imgidx_gd, code, args.weight_sc)

            elif model_type == 'srnn':
                seg_rc = srnet(seg_DE)
                loss_seg, loss_sr, loss = net_loss(seg_DE, cardiac_gd, seg_DE, seg_rc, args.weight_sr)

            elif model_type == 'srscn':
                code = scnet(encoding)
                seg_rc = srnet(seg_DE)
                loss_seg, loss_sc, loss_sr, loss = net_loss(seg_DE, cardiac_gd, imgidx_gd, code,
                                                            seg_DE, seg_rc, args.weight_sc, args.weight_sr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # write to log
            with open(f'logs/{model_type}_log_training.txt', 'a') as segment_log:
                segment_log.write("==> Epoch: {:0>3d}/{:0>3d} || ".format(epoch + 1, args.end_epoch))
                segment_log.write("Iteration: {:0>3d}/{:0>3d} - ".format(iteration + 1, IterCount))
                segment_log.write("LR: {:.6f} | ".format(float(optimizer.param_groups[0]['lr'])))
                if model_type == 'unet':
                    segment_log.write("loss_seg: {:.6f} \n".format(loss.detach().cpu()))
                elif model_type == 'scn':
                    segment_log.write("loss_seg: {:.6f} + ".format(loss_seg.detach().cpu()))
                    segment_log.write("loss_sc:{:.6f} + ".format(args.weight_sc * loss_sc.detach().cpu()))
                    segment_log.write("total_loss:{:.6f}\n".format(loss.detach().cpu()))
                elif model_type == 'srnn':
                    segment_log.write("loss_seg: {:.6f} + ".format(loss_seg.detach().cpu()))
                    segment_log.write("loss_sr:{:.6f} + ".format(args.weight_sr*loss_sr.detach().cpu()))
                    segment_log.write("total_loss:{:.6f}\n".format(loss.detach().cpu()))
                elif model_type == 'srscn':
                    segment_log.write("loss_seg: {:.6f} + ".format(loss_seg.detach().cpu()))
                    segment_log.write("loss_sc:{:.6f} + ".format(args.weight_sc * loss_sc.detach().cpu()))
                    segment_log.write("loss_sr:{:.6f} + ".format(args.weight_sr * loss_sr.detach().cpu()))
                    segment_log.write("total_loss:{:.6f}\n".format(loss.detach().cpu()))

            # write to tensorboard（add_scalar(名称, 值, 时间步step)）
            if model_type == 'unet':
                writer.add_scalar('seg loss', loss.detach().cpu(), epoch * (IterCount + 1) + iteration)
            elif model_type == 'scn':
                writer.add_scalar('seg loss', loss_seg.detach().cpu(), epoch * (IterCount + 1) + iteration)
                writer.add_scalar('sc loss', loss_sc.detach().cpu(), epoch * (IterCount + 1) + iteration)
                writer.add_scalar('total loss', loss.detach().cpu(), epoch * (IterCount + 1) + iteration)
            elif model_type == 'srnn':
                writer.add_scalar('seg loss', loss_seg.detach().cpu(), epoch * (IterCount + 1) + iteration)
                writer.add_scalar('sr loss', loss_sr.detach().cpu(), epoch * (IterCount + 1) + iteration)
                writer.add_scalar('total loss', loss.detach().cpu(), epoch * (IterCount + 1) + iteration)
            elif model_type == 'srscn':
                writer.add_scalar('seg loss', loss_seg.detach().cpu(), epoch * (IterCount + 1) + iteration)
                writer.add_scalar('sc loss', loss_sc.detach().cpu(), epoch * (IterCount + 1) + iteration)
                writer.add_scalar('sr loss', loss_sr.detach().cpu(), epoch * (IterCount + 1) + iteration)
                writer.add_scalar('total loss', loss.detach().cpu(), epoch * (IterCount + 1) + iteration)

        lr_scheduler.step()

        # Validation
        model.eval()
        avg_dice_2d = Validation2d(args, epoch, model, Valid_Image, Valid_loader, writer,
                                   f'logs/{model_type}_validation_2d.txt', tensorboardImage=True)

        # save the model if the average dice score exceeds a threshold
        if avg_dice_2d > args.threshold:
            torch.save(model.state_dict(),
                       Path('checkpoints').joinpath(model_type, str(avg_dice_2d) + '[' + str(epoch + 1) + '].pth'))
