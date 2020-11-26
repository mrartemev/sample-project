import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

from .inception import InceptionV3
from ..utils import one_hot


def get_activations(config, real_data, inception):
    inception.eval()
    activations = []

    for i in range(0, len(real_data), config.experiment.batch_size):

        batch = torch.from_numpy(real_data[i: i + config.experiment.batch_size]).float().to(config.utils.device)
        pred = inception(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        activations.append(pred.detach().cpu().view(pred.size(0), -1).numpy())
    activations = torch.cat(activations, dim=0)
    return activations


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(config, dataloader, model, inception):
    inception.eval()
    model.eval()
    real_activations, fake_activations = [], []

    for batch in tqdm(dataloader, desc='fid loop', leave=False):
        x, label = batch[0].to(config.utils.device), batch[1].to(config.utils.device)

        real = inception(x)[0]
        if real.size(2) != 1 or real.size(3) != 1:
            real = adaptive_avg_pool2d(real, output_size=(1, 1))
        real_activations.append(real.cpu().view(real.size(0), -1).numpy())

        fake = inception(model.generate(x, label))[0]
        if fake.size(2) != 1 or fake.size(3) != 1:
            fake = adaptive_avg_pool2d(fake, output_size=(1, 1))
        fake_activations.append(fake.cpu().view(real.size(0), -1).numpy())

    real_activations = np.concatenate(real_activations, axis=0)
    fake_activations = np.concatenate(fake_activations, axis=0)

    return np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False), \
           np.mean(fake_activations, axis=0), np.cov(fake_activations, rowvar=False),

@torch.no_grad()
def calculate_fid(config, model, dataloader, dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception = InceptionV3([block_idx], normalize_input=False).to(config.utils.device)
    model.to(config.utils.device)

    m1, s1, m2, s2 = calculate_activation_statistics(config, dataloader, model, inception)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value.item()
