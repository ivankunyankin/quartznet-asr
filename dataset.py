import re
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import TextTransform, augment


class LibriDataset(Dataset):

    def __init__(self, config, set):
        super(LibriDataset, self).__init__()

        self.config = config
        self.parameters = config[set]

        self.label_encoder = TextTransform()

        with open(self.parameters["data_list"], "r") as f:
            data = f.readlines()
        data = [line.strip().split() for line in data]

        self.collection = data

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, item):
        spec_path, transcript_path = self.collection[item]

        # read spectrograms
        melspec = torch.from_numpy(np.load(spec_path))

        # read and preprocess transcripts
        with open(transcript_path, "r") as text:
            transcript = text.read()
            transcript = transcript.lower()
            transcript = re.sub("[^'A-Za-z0-9 ]+", '', transcript)
            transcript = torch.tensor(self.label_encoder.text_to_int(transcript), dtype=torch.long)

        # apply time and frequency masking
        if self.parameters.get("apply_masking", None):
            melspec = augment(melspec, *self.config["masking"].values())

        input_length = melspec.shape[1]
        label_length = len(transcript)

        return melspec, transcript, input_length, label_length
