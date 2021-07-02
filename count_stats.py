import os
import glob
import yaml
import librosa
import argparse
import numpy as np

import torch
from utils import audio_to_mel


def main(folders, config, per_channel):

    print("globbing...")

    paths = []
    for folder in folders:
        paths.extend([path for path in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if path.endswith((".flac", ".wav"))])

    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_files = 0, 0, 0

    for path in paths:

        audio, _ = librosa.load(path, sr=config["spec_params"]["sr"])
        melspec = audio_to_mel(audio, config["spec_params"])

        if per_channel:
            channels_sum += torch.mean(melspec, dim=1)
            channels_sqrd_sum += torch.mean(melspec**2, dim=1)
        else:
            channels_sum += torch.mean(melspec)
            channels_sqrd_sum += torch.mean(melspec**2)

        num_files += 1

    mean = (channels_sum/num_files)
    std = (channels_sqrd_sum/num_files - mean**2)**0.5

    np.save(config["stats"], torch.cat([mean.unsqueeze(0), std.unsqueeze(0)], dim=0).unsqueeze(0).numpy())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--folders", nargs="+", type=str)
    parser.add_argument('--conf', default="config.yml", help='Path to the configuration file')
    parser.add_argument('--per_channel', default=False, action="store_true", help='Calculate per mel channel stats')

    args = parser.parse_args()

    config = yaml.safe_load(open(args.conf))
    folders = args.folders
    per_channel = args.per_channel

    main(folders, config, per_channel)
