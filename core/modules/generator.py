import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

log = logging.getLogger(__name__)


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config

        self.encoder, self.encoder_channels = self._build_encoder()
        self.decoder = self._build_decoder()
        self.refinement = self._build_refinement()

    def _build_encoder(self):
        # every encoder layer has 2 * previous layer channels.
        # every encoder layer downscale image by a factor of two.
        encoder = nn.ModuleList()
        # example: conv_dim = 64, num_layers = 3
        # encoder_channels = [64, 128, 256]
        encoder_channels = [self.config.model.G.conv_dim]
        # Using firstmost conv layer to let modules process color in the right way
        # see: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/782
        encoder.append(
            nn.Sequential(
                nn.Conv2d(3 + 3, self.config.model.G.conv_dim, 3, padding=1),
                nn.BatchNorm2d(self.config.model.G.conv_dim),
                nn.LeakyReLU(0.1),
                nn.Conv2d(self.config.model.G.conv_dim, self.config.model.G.conv_dim, 3, padding=1, stride=2),
                nn.BatchNorm2d(self.config.model.G.conv_dim),
                nn.LeakyReLU(0.1),
            ))
        log.info(f"Encoder 0: {3 + 3} -> {self.config.model.G.conv_dim} -> {self.config.model.G.conv_dim}")
        for i in range(1, self.config.model.G.num_layers):
            encoder_channels.append(min(encoder_channels[-1] * 2, self.config.model.max_features))
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(encoder_channels[-2], encoder_channels[-1], 3, padding=1, stride=2),
                    nn.BatchNorm2d(encoder_channels[-1]),
                    nn.LeakyReLU(0.1),
                )
            )
            log.info(f"Encoder {i}: {encoder_channels[-2]} -> {encoder_channels[-1]}")
        return encoder, encoder_channels

    def _build_decoder(self):
        decoder = nn.ModuleList()
        decoder.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(self.encoder_channels[-1] + self.config.experiment.n_atts,
                          self.encoder_channels[-1], 3, padding=1),
                nn.BatchNorm2d(self.encoder_channels[-1]),
                nn.LeakyReLU(0.1)
            )
        )
        log.info(f"Decoder 0: {self.encoder_channels[-1] + self.config.experiment.n_atts} -> {self.encoder_channels[-1]}")
        # every decoder layer upsample image by a factor of two.
        for dec_ind, (dec_dim, connection_dim) in enumerate(zip(self.encoder_channels[:0:-1], self.encoder_channels[-2::-1])):
            decoder.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(dec_dim + connection_dim, connection_dim, 3, padding=1),
                    nn.BatchNorm2d(connection_dim),
                    nn.LeakyReLU(0.1)
                )
            )
            log.info(f"Decoder {dec_ind + 1}: {dec_dim} + {connection_dim} -> {connection_dim}")
        decoder.append(
            nn.Sequential(
                nn.Conv2d(self.encoder_channels[0], self.config.model.G.conv_dim, 3, padding=1),
                nn.BatchNorm2d(self.config.model.G.conv_dim),
                nn.LeakyReLU(0.1)
            )
        )
        log.info(f"Decoder last: {self.encoder_channels[0]} -> {self.config.model.G.conv_dim}")
        return decoder

    def _build_refinement(self):
        return nn.Sequential(
            nn.Conv2d(self.config.model.G.conv_dim + self.config.experiment.n_atts + 3,
                      self.config.model.G.conv_dim, 3, padding=1),
            nn.BatchNorm2d(self.config.model.G.conv_dim),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.config.model.G.conv_dim, self.config.model.G.conv_dim, 3, padding=1),
            nn.BatchNorm2d(self.config.model.G.conv_dim),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.config.model.G.conv_dim, 3, 1),
        )

    def forward(self, img_a, a, reduce_noise=False):
        # propagate encoder layers
        if reduce_noise:
            x = torch.cat([img_a, torch.randn_like(img_a) * 0.5], dim=1)
        else:
            x = torch.cat([img_a, torch.randn_like(img_a)], dim=1)
        x = self.encoder[0](x)
        layer_results = [x]

        for layer in self.encoder[1:]:
            x = layer(x)
            layer_results.append(x)

        # expanding and concating attributes to encoder output
        out = layer_results[-1]

        n, _, h, w = out.size()
        attr = a.view((n, self.config.experiment.n_atts, 1, 1)).expand((n, self.config.experiment.n_atts, h, w))
        # passing attributes an encoder output through decoder's first layer
        out = self.decoder[0](torch.cat([out, attr], dim=1))

        # propagate decoder layers
        for encoder_out, decoder in zip(layer_results[-2::-1], self.decoder[1:-1]):
            out = F.interpolate(out, (encoder_out.shape[2], encoder_out.shape[3]))
            out = torch.cat([out, encoder_out], dim=1)
            out = decoder(out)
        out = F.interpolate(out, (img_a.shape[2], img_a.shape[3]))
        out = self.decoder[-1](out)

        n, _, h, w = out.size()
        attr = a.view((n, self.config.experiment.n_atts, 1, 1)).expand((n, self.config.experiment.n_atts, h, w))

        out = self.refinement(torch.cat([out, img_a, attr], dim=1))
        out = torch.tanh(out)
        return out
