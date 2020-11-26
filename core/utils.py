import torch


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def _getattr(self, key):
        target = self
        for dot in key.split('.'):
            target = target[dot]
        return target

    def _setattr(self, key, value):
        target = self
        for dot in key.split('.')[:-1]:
            target = target[dot]
        target[key.split('.')[-1]] = value


class InfiniteDataloader:
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def get_next(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            return self.get_next()


class DDPWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module.module.forward(*args, **kwargs)

    def generate_mask(self, *args, **kwargs):
        return self.module.module.generate_mask(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.module.module.generate(*args, **kwargs)


def one_hot(x, num_classes):
    y = torch.zeros(x.shape[0], num_classes)
    y[range(y.shape[0]), x] = 1
    return y


def mask2float(mask):
    if mask.shape[1] == 1:
        mask = mask.repeat(1, 3, 1, 1)
    return (mask.float() - 0.5) * 2


def to_grayscale(img):
    return (img[:, 0] * 0.2989 + img[:, 1] * 0.5870 + img[:, 2] * 0.1140).unsqueeze(1)
