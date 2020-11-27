import os
import numpy as np
from tqdm import tqdm
import wandb

import torch
import torchvision.utils as vutils
import torch.distributed as dist

import torch.utils.data
from .data import get_loaders
from .metrics import calculate_fid
from .trainer import Trainer
import logging
from PIL import Image
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def setup_experiment(config):
    wandb.login()
    wandb.init(project="sample-project",
               config=OmegaConf.to_container(config, resolve=True))


def train(gpu_num_if_use_ddp, config):
    # comet.ml will automatically upload metrics to it own server. To watch it, find a comet.ml link in logs
    # By default it will collect a lot of different stuff, since our project is somewhat NDA, we shouldn't share it.
    if config.utils.use_ddp:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:4433",
            world_size=torch.cuda.device_count(),
            rank=gpu_num_if_use_ddp,
        )
        config.utils.device = f'cuda:{gpu_num_if_use_ddp}'
        main_node = gpu_num_if_use_ddp == 0
    else:
        main_node = True
    if main_node:
        setup_experiment(config)
    train_dataloader, test_dataloader, val_dataloader = get_loaders(config)

    trainer = Trainer(config)

    try:
        trainer.load(config.experiment.model.checkpoint_path)
        log.info(f'Loaded checkpoint: {config.experiment.model.checkpoint_path} successfully')
    except Exception as e:
        log.info(f"{e}\nCan't load checkpoint: {config.experiment.model.checkpoint_path}. Started from zeroth epoch")

    if main_node:
        wandb.watch(trainer.model, log="all")

    def prepare_batch(batch):
        img_a, att_a = batch
        img_a, att_a = img_a.to(config.utils.device), att_a.to(config.utils.device).float()
        return img_a, att_a

    # fixing images and attributes for the sampling procedure
    fixed_img_a, fixed_att_a = prepare_batch(next(iter(val_dataloader)))

    # list for sampling during eval part.
    sample_att_b_list = []
    for i in range(config.experiment.n_atts):
        tmp = torch.zeros_like(fixed_att_a)
        tmp[:, i] = 1
        sample_att_b_list.append(tmp)

    for epoch in range(0, config.experiment.epochs):
        log.info(f"Epoch {epoch} started")
        trainer.model.train()
        for iteration in tqdm(range(config.utils.epoch_iters), desc='train loop', leave=False, position=0):
            img_a, att_a = prepare_batch(next(train_dataloader))
            if iteration == 0:
                print(f"First batch of epoch {epoch} with shapes:"
                         f" image {img_a.shape},  att_a {att_a.shape}."
                         f" Working on {config.utils.device}")
                print(f"First batch of epoch {epoch} with means:"
                         f" image {img_a.mean()}, att_a {att_a.mean()}."
                         f" Working on {config.utils.device}")
                print(f"First batch of epoch {epoch} with stds:"
                         f" image {img_a.std()}, att_a {att_a.std()}."
                         f" Working on {config.utils.device}")

            d_loss = trainer.train('D', img_a, att_a)
            g_loss = trainer.train('G', img_a, att_a)
            # logging metrics every N iterations to save some time on uploads
            if (iteration + 1) % config.utils.log_iter_interval == 0 and main_node:
                trainer.model.eval()
                wandb.log(trainer.evaluate('D', img_a, att_a, tag='training'))
                wandb.log(trainer.evaluate('G', img_a, att_a, tag='training'))
                trainer.model.train()

        if (epoch + 1) % config.utils.save_interval == 0 and main_node:
            trainer.save(os.path.join(os.getcwd(), 'checkpoint', 'weights.{:d}.pth'.format(epoch)))
            log.info(f"Model checkpointed to {os.path.join(os.getcwd(), 'checkpoint', 'weights.{:d}.pth'.format(epoch))}")
            # checkpoint_fid = calculate_fid(config, trainer.model, test_dataloader, dims=2048)
            # wandb.log('checkpoint/FID', checkpoint_fid)

        if (epoch + 1) % config.utils.sample_interval == 0 and main_node:
            trainer.model.eval()
            with torch.no_grad():
                samples = [fixed_img_a]
                for i, att_b in enumerate(sample_att_b_list):
                    samples.append(trainer.model.generate(fixed_img_a, att_b))
                samples = torch.cat(samples, dim=3)

                output_path = os.path.join(os.getcwd(), 'sample_training', f'Epoch_{epoch}.jpg')
                vutils.save_image(samples, output_path, normalize=True, range=(-1, 1), nrow=1)
                wandb.log({"sample_training": [wandb.Image(Image.open(output_path), caption=f"epoch_{epoch}")]})

        if (epoch + 1) % config.utils.eval_interval == 0 and main_node:
            lossD, lossG = [], []
            trainer.model.eval()
            for batch in tqdm(test_dataloader, desc='val_loop', leave=False, position=0):
                img_a, att_a = prepare_batch(batch)
                with torch.no_grad():
                    lossD.append(trainer.evaluate('D', img_a, att_a, tag='evaluation'))
                    lossG.append(trainer.evaluate('G', img_a, att_a, tag='evaluation'))
            wandb.log({key: np.mean([i[key] for i in lossD]) for key in lossD[0].keys()})
            wandb.log({key: np.mean([i[key] for i in lossD]) for key in lossG[0].keys()})
            log.info(f"Validation after epoch {epoch} finished")
    log.info("Training finished")
