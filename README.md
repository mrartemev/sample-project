# Сeleba GAN, PyTorch.

Here we use a simple pipeline to train GAN to change faces accordion to the Celeba dataset.

We use a model architecture similar to AttGAN, having a simple generator and a critic with classifier.
On each iteration additionally to WGAN losses there are a classification loss for both gen/critic and reco loss for gen.

#### Usage

1. Build conda environment:
    `conda env create -f environment.yml`
2. Login to wandb:
    `wandb login`
2. Run:
    `python main.py [args]`
3. Check wandb logs

#### Config

Config in mainteined by [hydra](hydra.cc). 
Please read the quick guide for clarity


### Additional info

One can use a DDP setup by using *utils.use_ddp=1*. In this case there will be create N processes, one for each GPU.

Here we decided to use a simple pytorch dataloader, without `torch.utils.data.DistributedSampler`.
Because of that, the overall time would not decrease by using DDP, rather the batch size would be virtually increased