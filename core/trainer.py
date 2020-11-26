import torch

import torch.optim as optim
from .model import HairRecolorGAN
from .utils import DDPWrapper


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = HairRecolorGAN(config).to(config.utils.device)
        self.optim = {'G': optim.Adam(self.model.G.parameters(), lr=config.experiment.lr.G, betas=(0.5, 0.9)),
                      'D': optim.Adam(self.model.D.parameters(), lr=config.experiment.lr.D, betas=(0.5, 0.9))}
        self.names = {'G': self.model.G, 'D': self.model.D}
        if self.config.utils.use_ddp:
            self.move_to_ddp(self.config.utils.device)

    def train(self, module, img_a, att_a):
        self.optim[module].zero_grad()
        loss = self.model(module, img_a, att_a, mode='train')
        loss.backward()
        if self.config.experiment.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.names[module].parameters(), self.config.experiment.grad_clip)
        self.optim[module].step()
        return loss.item()

    def evaluate(self, module, img_a, att_a, tag='training'):
        return self.model(module, img_a, att_a, mode=tag)

    def move_to_ddp(self, device_id):
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[device_id], find_unused_parameters=True)
        self.model = DDPWrapper(self.model)

    def save(self, path):
        model = self.model.module.module if self.config.utils.use_ddp else self.model
        states = {
            'G': model.G.state_dict(),
            'D': model.D.state_dict(),
            'optim_G': self.optim['G'].state_dict(),
            'optim_D': self.optim['D'].state_dict(),
        }
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        model = self.model.module.module if self.config.utils.use_ddp else self.model
        if 'G' in states:
            model.G.load_state_dict(states['G'])
        if 'D' in states:
            model.D.load_state_dict(states['D'])
        if 'optim_G' in states:
            self.optim['G'].load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim['D'].load_state_dict(states['optim_D'])
