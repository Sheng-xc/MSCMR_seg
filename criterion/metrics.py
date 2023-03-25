class Criterion(object):
    def calc_Dice(self, output, target):
        num = (output * target).sum()
        den = output.sum() + target.sum()
        dice = 2 * num / (den + 1e-8)
        return dice

    def __call__(self, output, target):

        output = output.numpy()
        target = target.numpy()

        myo_dice = self.calc_Dice((output == 1), (target == 1))  # myo
        lv_dice = self.calc_Dice((output == 2), (target == 2))   # lv
        rv_dice = self.calc_Dice((output == 3), (target == 3))   # rv
        return myo_dice, lv_dice, rv_dice


class DiceMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.myo = {'sum': 0.0, 'avg': 0.0, 'list': []}
        self.lv = {'sum': 0.0, 'avg': 0.0, 'list': []}
        self.rv = {'sum': 0.0, 'avg': 0.0, 'list': []}
        self.count = 0

    def update(self, myo_dice, lv_dice, rv_dice):
        self.count += 1
        self.myo['sum'] += myo_dice
        self.myo['avg'] = self.myo['sum'] / self.count
        self.myo['list'].append(myo_dice)
        self.lv['sum'] += lv_dice
        self.lv['avg'] = self.lv['sum'] / self.count
        self.lv['list'].append(lv_dice)
        self.rv['sum'] += rv_dice
        self.rv['avg'] = self.rv['sum'] / self.count
        self.rv['list'].append(rv_dice)


class Logger2d(object):
    def __call__(self, epoch, total_epoch, file_name, DE_dice):
        with open(file_name, 'a') as f:
            f.write("=> Epoch: {:0>3d}/{:0>3d} || ".format(epoch, total_epoch))
            f.write("DE(myo,lv,rv): {:.4f} - {:.4f} - {:.4f} + ".format(DE_dice['myo'], DE_dice['lv'], DE_dice['rv']))
            f.write("Avg: {:.4f}\n".format((DE_dice['myo'] + DE_dice['lv'] +  DE_dice['rv']) / 3))
