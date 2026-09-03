# import comet_ml
import hydra
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.datasets import BaseDataset, collate_fn


class LogMelSpectrogram:
    def __init__(
        self,
        sample_rate=16000,
        n_mels=80,
    ):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
        )

    def __call__(self, data):
        # data: waveform [T]

        spec = self.mel(data)

        # log scale
        spec = torch.log(spec + 1e-9)

        return spec


# @hydra.main(version_base=None, config_path="src/configs", config_name="conf")
def train() -> None:
    # DATASETS
    transform = LogMelSpectrogram(sample_rate=16000, n_mels=80)

    train_dataset = BaseDataset(
        path_data_dir="data/datasets/librispeech/dev-clean", transforms=transform
    )
    # DATALOADERS
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=12,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    print("DSDSs")
    for bath in train_dataloader:
        print(bath)
        break


train()
