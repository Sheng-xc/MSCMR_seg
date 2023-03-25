import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import warnings

warnings.filterwarnings("ignore")

from utils import config
from itertools import cycle
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import weights_init, DEDataLoader
from network import ConSR


def SRTrain(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ConSR(4).to(device)
    model.apply(weights_init)

    net_loss = nn.MSELoss(reduction='sum')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6, last_epoch=-1)

    if not Path.exists(Path('checkpoints')):
        Path('checkpoints').mkdir()

    if not Path.exists(Path('logs')):
        Path('logs').mkdir()

    Train_Image = DEDataLoader(path=args.path, file_name='train.txt', dim=args.dim,
                               max_iters=args.batch_size, stage='Train')
    Train_loader = cycle(
        DataLoader(Train_Image, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True))

    Valid_Image = DEDataLoader(path=args.path, file_name='valid.txt', dim=args.dim, max_iters=None, stage='Train')
    Valid_loader = cycle(DataLoader(Valid_Image, batch_size=1, shuffle=False, num_workers=0, drop_last=False))

    writer = SummaryWriter('logs/pretrain')
    for epoch in range(args.start_epoch, args.end_epoch):
        # train
        model.train()
        cardiac_gd = torch.FloatTensor(args.batch_size, 4, args.dim, args.dim).cuda()

        IterCount = int(len(Train_Image) / args.batch_size)  # of Iterations for each epoch
        for iteration in range(IterCount):
            _, label_cardiac, _ = next(Train_loader)

            # model output
            cardiac_gd.copy_(label_cardiac)
            cardiac_rc = model(cardiac_gd)
            loss = net_loss(cardiac_rc, cardiac_gd)

            # optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # write to tensorboard
            writer.add_scalar('reconstruction loss', loss.detach().cpu(), epoch * (IterCount + 1) + iteration)

        lr_scheduler.step()

        # validation
        model.eval()
        for iter in range(int(len(Valid_Image))):
            _, label_val, _ = next(Valid_loader)

            cardiac_gd.copy_(label_val)
            cardiac_rc = model(cardiac_gd)
            loss = net_loss(cardiac_rc, cardiac_gd)
            writer.add_scalar('validation loss', loss.detach().cpu(), epoch * (IterCount + 1) + iter)

    torch.save(model.state_dict(), Path('checkpoints') / 'sc.pth')


if __name__ == '__main__':
    args = config()
    SRTrain(args)
