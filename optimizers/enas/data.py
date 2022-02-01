from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def get_loaders(args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD,
        ),
    ])
    train_dataset = CIFAR10(
        root=args.data,
        train=True,
        download=True,
        transform=train_transform,
    )

    indices = list(range(len(train_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(indices[:-5000]),
        pin_memory=True,
        num_workers=2,
    )

    reward_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(indices[-5000:]),
        pin_memory=True,
        num_workers=2,
    )

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD,
        ),
    ])
    valid_dataset = CIFAR10(
        root=args.data,
        train=False,
        download=False,
        transform=valid_transform,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    # repeat_train_loader = RepeatedDataLoader(train_loader)
    repeat_reward_loader = RepeatedDataLoader(reward_loader)
    repeat_valid_loader = RepeatedDataLoader(valid_loader)

    return train_loader, repeat_reward_loader, repeat_valid_loader


class RepeatedDataLoader():
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = self.data_loader.__iter__()

    def __len__(self):
        return len(self.data_loader)

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()
        return batch
