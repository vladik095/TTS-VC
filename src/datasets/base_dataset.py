import os

import torchaudio
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, path_data_dir, transforms=None):
        self.data_index = []

        self._build_data_index(path_data_dir)
        self.transforms = transforms

    def __getitem__(self, index):
        data_dict = self.data_index[index]

        text = data_dict["text"]
        audio_path = data_dict["path_audio"]

        audio = self.get_audio(audio_path)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = spectrogram.squeeze(0).transpose(0, 1)

        data_obj = {
            "text": text,
            "spectrogram": spectrogram,
            "audio_path": audio_path,
        }
        return data_obj

    def __len__(self):
        return len(self.data_index)

    def get_spectrogram(self, audio):
        mel_spectrogram = self.transforms(audio)

        return mel_spectrogram

    def get_audio(self, path_audio):
        waveform, _ = torchaudio.load(path_audio)
        return waveform

    def _build_data_index(self, path_data_dir):
        for dirname, _, filenames in os.walk(path_data_dir):
            for filename in filenames:
                full_path = os.path.join(dirname, filename)

                if filename.endswith("trans.txt"):
                    with open(full_path, "r", encoding="utf-8") as file:
                        for line in file:
                            line = line.strip()
                            audio_id, text = line.split(maxsplit=1)

                            audio_path = (
                                os.path.dirname(full_path) + f"/{audio_id}.flac"
                            )

                            exmp = {"path_audio": audio_path, "text": text.lower()}
                            self.data_index.append(exmp)
