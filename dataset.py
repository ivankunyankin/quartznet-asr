import re
import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from utils import TextTransform, audio_to_mel, augment


class LibriDataset(Dataset):

    def __init__(self, config, set):
        super(LibriDataset, self).__init__()

        self.config = config
        self.parameters = config[set]

        self.label_encoder = TextTransform()

        if not os.path.exists(self.parameters["data_list"]):
            self.create_data_list()

        if self.config["normalize"]:
            if os.path.exists(self.config["stats"]):
                stats = torch.from_numpy(np.load(self.config["stats"]))
                if stats.shape[0] == 1:
                    self.mean = stats[0, 0]
                    self.std = stats[0, 1]
                else:
                    self.mean = stats[:, 0].unsqueeze(1)
                    self.std = stats[:, 1].unsqueeze(1)

        with open(self.parameters["data_list"], "r") as f:
            data = f.readlines()
        data = [line.strip().split() for line in data]

        self.collection = data

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, item):
        audio, transcript = self.collection[item]

        audio, _ = librosa.load(audio, sr=self.config["spec_params"]["sr"])
        with open(transcript, "r") as text:
            transcript = text.read()
            transcript = transcript.lower()
            transcript = re.sub("[^'A-Za-z0-9 ]+", '', transcript)
            transcript = torch.tensor(self.label_encoder.text_to_int(transcript), dtype=torch.long)

        # apply time stretch
        if self.parameters.get("apply_speed_pertrubation", None):
            limit = self.config.get("speed_pertrubation", 0.1)
            rate = np.random.uniform(low=1-limit, high=1+limit)
            audio = librosa.effects.time_stretch(audio, rate)

        # generate mel spectrogram
        melspec = audio_to_mel(audio, self.config["spec_params"])

        # apply normalization
        if self.config["normalize"]:
            melspec = (melspec - self.mean) / self.std

        # apply time and frequency masking
        if self.parameters.get("apply_masking", None):
            melspec = augment(melspec, *self.config["masking"].values())

        input_length = melspec.shape[1]
        label_length = len(transcript)

        return melspec, transcript, input_length, label_length

    def create_data_list(self):

        data_list = open(self.parameters["data_list"], "w")

        for folder in self.parameters["data_dirs"]:
            for root, dirs, files in os.walk(folder):
                for file in files:

                    if file.endswith((".flac", ".wav")):

                        audio, sr = librosa.load(os.path.join(root, file), sr=self.config["spec_params"]["sr"])

                        # length check
                        if self.config.get("max_length", None):
                            length = audio.shape[0] / sr
                            if length > int(self.config["max_length"]):
                                continue

                        # label check
                        label = os.path.splitext(file)[0] + ".normalized.txt"
                        if not os.path.exists(os.path.join(root, label)):
                            continue

                        data_list.write(f"{os.path.join(root, file)} {os.path.join(root, label)} {os.linesep}")

        data_list.close()
