import re
import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from utils import TextTransform, audio_to_mel, augment


class LibriDataset(Dataset):

    def __init__(self, config, model, set, cash):
        super(LibriDataset, self).__init__()

        self.cash = cash
        self.config = config
        self.parameters = config[model][set]

        self.label_encoder = TextTransform()

        if not os.path.exists(self.parameters["data_list"]):
            self.create_data_list()

        with open(self.parameters["data_list"], "r") as f:
            data = f.readlines()
        data = [line.strip().split() for line in data]

        if self.cash:

            self.collection = []

            for sample in data:
                audio, transcript = sample
                audio, _ = librosa.load(audio, sr=self.config["spec_params"]["sr"])
                with open(transcript, "r") as text:
                    transcript = text.read()
                transcript = transcript.lower()
                transcript = re.sub("[^'A-Za-z0-9 ]+", '', transcript)
                transcript = torch.tensor(self.label_encoder.text_to_int(transcript), dtype=torch.long)

                self.collection.append([audio, transcript])
        else:
            self.collection = data

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, item):
        audio, transcript = self.collection[item]

        if not self.cash:
            audio, _ = librosa.load(audio, sr=self.config["spec_params"]["sr"])
            with open(transcript, "r") as text:
                transcript = text.read()
                transcript = transcript.lower()
                transcript = re.sub("[^'A-Za-z0-9 ]+", '', transcript)
                transcript = torch.tensor(self.label_encoder.text_to_int(transcript), dtype=torch.long)

        # apply time stretch
        if self.parameters.get("speed_pertrubation", None):
            rate = np.random.uniform(low=0.9, high=1.1)
            audio = librosa.effects.time_stretch(audio, rate)

        # apply per mel channel normalization
        if os.path.exists(self.config["channel_norm"]):
            melspec = torch.from_numpy(audio_to_mel(audio, self.config["spec_params"], apply_normalize_spect=False))
            stats = torch.from_numpy(np.load(self.config["channel_norm"]))
            mean = stats[:, 0].unsqueeze(1)
            std = stats[:, 1].unsqueeze(1)
            melspec = (melspec - mean) / std
        else:
            print("\n* Stats file was not found. Applying rough normalization\n")
            melspec = torch.from_numpy(audio_to_mel(audio, self.config["spec_params"]))

        # apply time and frequency masking
        if self.parameters.get("masking", None):
            melspec = augment(melspec)

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
                        if self.parameters.get("max_length", None):
                            length = audio.shape[0] / sr
                            if length > int(self.parameters["max_length"]):
                                continue

                        # label check
                        label = os.path.splitext(file)[0] + ".transcription.txt"
                        if not os.path.exists(os.path.join(root, label)):
                            continue

                        data_list.write(f"{os.path.join(root, file)} {os.path.join(root, label)} {os.linesep}")

        data_list.close()
