import os
import yaml
import librosa
import argparse
import numpy as np

from utils import audio_to_mel


def main(config):

    sets = ["train", "val", "test"]

    for set_ in sets:

        print(f"=> Preparing {set_} set...")

        num_files = 0
        num_valid = 0

        data_list = open(config[set_]["data_list"], "w")

        for folder in config[set_]["data_dirs"]:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith((".flac", ".wav")):
                        num_files += 1

                        audio, sr = librosa.load(os.path.join(root, file), sr=config["spec_params"]["sr"])

                        # length check
                        if config.get("max_length", None):
                            length = audio.shape[0] / sr
                            if length > int(config["max_length"]):
                                continue

                        # label check
                        label = os.path.splitext(file)[0] + ".normalized.txt"
                        if not os.path.exists(os.path.join(root, label)):
                            continue

                        melspec = audio_to_mel(audio, config["spec_params"])

                        # save spectrogram
                        spec_name = f"{'.'.join(file.split('.')[:-1])}.npy"
                        np.save(os.path.join(root, spec_name), melspec)

                        data_list.write(f"{os.path.join(root, spec_name)} {os.path.join(root, label)} {os.linesep}")
                        num_valid += 1

        data_list.close()
        print(f"...overall files: {num_files}")
        print(f"...valid files: {num_valid}")
    print("=> Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--conf', default="config.yml", help='Path to the configuration file')

    args = parser.parse_args()
    config = yaml.safe_load(open(args.conf))
    main(config)