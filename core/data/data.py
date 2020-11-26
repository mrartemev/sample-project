import torch.utils.data as data
import torchvision
from torchvision.transforms import ToTensor, Normalize, Resize, Compose


def get_transforms(config):
    return Compose([
        Resize((config.model.img_size, config.model.img_size)),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def get_datasets(config):
    train_dataset = torchvision.datasets.CelebA(root=config.data.data_path,
                                                split='train',
                                                target_type='attr',
                                                transform=get_transforms(config),
                                                target_transform=None,
                                                download=config.data.download)
    test_dataset = torchvision.datasets.CelebA(root=config.data.data_path,
                                                split='test',
                                                target_type='attr',
                                                transform=get_transforms(config),
                                                target_transform=None,
                                                download=config.data.download)
    val_dataset = torchvision.datasets.CelebA(root=config.data.data_path,
                                                split='valid',
                                                target_type='attr',
                                                transform=get_transforms(config),
                                                target_transform=None,
                                                download=config.data.download)
    return train_dataset, test_dataset, val_dataset


def get_loaders(config):
    train_dataset, test_dataset, val_dataset = get_datasets(config)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config.experiment.batch_size,
        num_workers=config.utils.num_workers,
        sampler=data.DistributedSampler(train_dataset) if config.utils.use_ddp else None,
        pin_memory=True,
        drop_last=True
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=config.experiment.batch_size,
        sampler=None,
        num_workers=config.utils.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        sampler=None,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, test_loader, val_loader
