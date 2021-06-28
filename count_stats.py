import os
import glob
import yaml
import librosa
import argparse
import numpy as np

import torch
from audio_to_mel import audio_to_spect


def main(folders, config):

    print("globbing...")

    paths = []
    for folder in folders:
        paths.extend([path for path in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if path.endswith((".flac", ".wav"))])

    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_files = 0, 0, 0

    for path in paths:

        audio, _ = librosa.load(path, sr=config["spec_params"]["sr"])
        melspec = torch.from_numpy(audio_to_spect(audio, config["spec_params"], apply_normalize_spect=False))

        channels_sum += torch.mean(melspec, dim=1)
        channels_sqrd_sum += torch.mean(melspec**2, dim=1)
        num_files += 1

    mean = (channels_sum/num_files)
    std = (channels_sqrd_sum/num_files - mean**2)**0.5

    np.save("stats.npy", torch.cat([mean.unsqueeze(1), std.unsqueeze(1)], dim=1).numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--conf', default="config.yml", help='Path to the configuration file')
    parser.add_argument("--folders", nargs="+", type=str)

    args = parser.parse_args()

    config = yaml.safe_load(open(args.conf))
    folders = args.folders

    main(folders, config)
