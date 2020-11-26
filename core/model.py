import torch

from .losses.discrimination import calculate_gradient_penalty
import torch.nn.functional as F
import logging

from .modules import Generator
from .modules import Discriminator
from .utils import DotDict

log = logging.getLogger(__name__)


class LossWeighter:
    def __init__(self, config):
        self.config = config
        self.weights = DotDict(config.experiment.losses)

    def combine_losses(self, pairs):
        combined_loss = 0
        for loss_name, loss in pairs.items():
            combined_loss += self.weights._getattr(loss_name) * loss
        return combined_loss

    def visualize_losses(self, pairs, mode):
        visualized_losses = {}
        for loss_name, loss in pairs.items():
            text = f"{mode}/{'/'.join(loss_name.split('.'))}"
            visualized_losses[text] = loss.item()
        return visualized_losses


class HairRecolorGAN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lw = LossWeighter(config)

        self.G = Generator(config)
        self.D = Discriminator(config)

    # here we need forward for ddp to activate forward hook
    def forward(self, module, *args, **kwargs):
        if module == 'G':
            return self.trainG(*args, **kwargs)
        if module == 'D' or module == 'C':
            return self.trainD(*args, **kwargs)

    def trainD(self, img_a, att_a, mode='training'):
        losses = {}

        att_b = att_a[torch.randperm(att_a.size(0))].to(self.config.utils.device)
        out_real, out_cls = self.D(img_a)
        # classification loss
        losses['D.classification'] = F.binary_cross_entropy_with_logits(out_cls, att_a)

        # compute critic with fake images
        img_b = self.G(img_a, att_b).detach()
        out_fake, out_cls = self.D(img_b)

        losses['D.adversarial'] = torch.mean(out_fake) - torch.mean(out_real)

        if mode == 'train':
            losses['D.gradient_penalty'] = calculate_gradient_penalty(self.D, img_a, img_b)

        if mode == 'train':
            return self.lw.combine_losses(losses)
        return self.lw.visualize_losses(losses, mode)

    def trainG(self, img_a, att_a, mode='training'):
        losses = {}

        att_b = att_a[torch.randperm(att_a.size(0))].clone()

        # compute critic with fake images
        img_b = self.G(img_a, att_b)
        out_fake, out_cls = self.D(img_b)

        # adversarial loss
        losses['G.adversarial'] = -torch.mean(out_fake)

        # classification loss
        # argmax to get label indexes
        losses['G.classification'] = F.binary_cross_entropy_with_logits(out_cls, att_b, reduction='mean')

        # target-to-original domain, reconstruction loss
        img_reco = self.G(img_b, att_a)
        losses['G.reconstruction'] = F.smooth_l1_loss(img_a, img_reco, reduction='mean')

        if mode == 'train':
            return self.lw.combine_losses(losses)
        return self.lw.visualize_losses(losses, mode)

    @torch.no_grad()
    def generate(self, img_a, att_b):
        img_b = self.G(img_a, att_b, reduce_noise=True)
        return img_b
