from torch.utils.data import DataLoader

from src.datasets.base_dataset import BaseDataset
from src.datasets.collate import collate_fn


def get_dataloader(config, transform, data_path):
    dataset = BaseDataset(path_data_dir=data_path, transforms=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=12,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    return dataloader
