import torch.nn as nn
import logging

log = logging.getLogger(__name__)


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.conv, conv_channels = self._build_conv_module(features_in=3)

        fc_in_features = (config.model.img_size // (2 ** (config.model.D.num_layers + 1))) ** 2
        self.fc = nn.Sequential(
            nn.Linear(conv_channels * fc_in_features, config.model.D.fc_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(config.model.D.fc_dim, 1 + self.config.experiment.n_atts)
        )

    def _build_conv_module(self, features_in):
        conv_dim = self.config.model.D.conv_dim
        layers = [
            nn.Sequential(
                    nn.Conv2d(features_in, conv_dim, 3, padding=1, stride=2),
                    nn.InstanceNorm2d(conv_dim),
                    nn.LeakyReLU(0.1)
            )
        ]
        in_channels = conv_dim
        for i in range(self.config.model.D.num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, 3, padding=1, stride=2),
                    nn.InstanceNorm2d(conv_dim),
                    nn.LeakyReLU(0.1)
                )
            )
            in_channels = conv_dim
            conv_dim = min(conv_dim * 2 ** (i + 1), self.config.model.max_features)
        conv = nn.Sequential(*layers)
        return conv, in_channels

    def forward(self, x):
        x = self.conv(x).view(x.shape[0], -1)
        x = self.fc(x)
        logit_adv, logit_att = x[:, 0], x[:, 1:]
        return logit_adv, logit_att
