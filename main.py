from os.path import join
import os
from core.train import train
import hydra
import logging
import torch
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"

def prepare_config(config):
    # modifying path to data
    if not os.path.isabs(config.data.data_path):
        config.data.data_path = os.path.join(hydra.utils.get_original_cwd(),
                                             config.data.data_path)
        log.debug(f"config.data.data_path modified to {config.data.data_path}")

    # modifying path to checkpoints
    if not os.path.isabs(config.experiment.model.checkpoint_path):
        config.experiment.model.checkpoint_path = os.path.join(hydra.utils.get_original_cwd(),
                                                               config.experiment.model.checkpoint_path)
        log.debug(f"config.experiment.model.checkpoint_path modified to {config.experiment.model.checkpoint_path}")


def prepare_dirs():
    # since hydra will run script in a new subfolder it is necessary to create dirs
    os.makedirs(join(os.getcwd(), 'checkpoint'), exist_ok=True)
    os.makedirs(join(os.getcwd(), 'sample_training'), exist_ok=True)
    os.makedirs(join(os.getcwd(), 'sample_masks'), exist_ok=True)
    log.debug("Created checkpoint, sample_training dirs")


# decorator allows hydra to load and parse config
# additionally, hydra will setup a global logger
@hydra.main(config_path=CONFIG_PATH)
def main(config):
    log.info(config.pretty())
    log.info("Current working directory  : {}".format(os.getcwd()))

    prepare_dirs()
    prepare_config(config)

    log.debug("Started training")
    # config = OmegaConf.to_container(config, resolve=True)
    if config.utils.use_ddp:
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(train, nprocs=world_size, args=(config,))
    train(None, config)


if __name__ == '__main__':
    main()
