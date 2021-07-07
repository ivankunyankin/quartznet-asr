import os
import yaml
import torch
import random
import datetime
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from jiwer import wer
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from IPython.display import clear_output
from torch.utils.data import DataLoader

from dataset import LibriDataset
from utils import TextTransform, custom_collate, create_model


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class Tester:

    def __init__(self, config):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Parameters
        hparams = config["encoder"]
        self.batch_size = hparams["batch_size"]
        self.weights = hparams["test"]["weights"]

        # Data
        self.dataset = LibriDataset(config, "test")
        self.loader = self.loader(self.dataset)
        self.processor = TextTransform()

        # Model
        self.model = create_model(
            model=config["model"],
            in_channels=config["spec_params"]["n_mels"],
            out_channels=len(self.processor.char_map)
        )
        self.model.to(self.device)
        self.model.module.load_state_dict(torch.load(self.weights, map_location=self.device))
        self.criterion = nn.CTCLoss(blank=28)

        # Logging
        now = datetime.datetime.now()
        path = os.path.join(hparams["log_dir"], now.strftime("%Y:%m:%d_%H:%M:%S"))
        self.writer = SummaryWriter(os.path.join(path, "test"))

    def test(self):

        self.model.eval()
        loop = tqdm(self.loader)
        losses = 0
        wers = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (specs, transcripts, input_lengths, label_length, _) in enumerate(loop):

                clear_output(wait=True)
                loop.set_description(f"Test")

                specs = specs.to(self.device)
                transcripts = transcripts.to(self.device)
                input_lengths = input_lengths.to(self.device)
                label_length = label_length.to(self.device)

                with torch.cuda.amp.autocast():
                    output = self.model(specs)
                    output = output.permute(2, 0, 1)
                    output = F.log_softmax(output, dim=2)
                    loss = self.criterion(output, transcripts, input_lengths, label_length)
                losses += loss

                loop.set_postfix(loss=loss.item())

                num_batches += 1

                decoded_preds, decoded_targets = self.processor.decode(output.permute(1, 0, 2), transcripts, label_length)
                error = wer(decoded_targets, decoded_preds)
                wers += error

                # Save training logs to Tensorboard
                rand_idx = random.randint(0, specs.shape[0] - 1)
                self.writer.add_text("predicted texts", decoded_preds[rand_idx], global_step=batch_idx)
                self.writer.add_text("target texts", decoded_targets[rand_idx], global_step=batch_idx)
                self.writer.add_scalar("loss", loss, global_step=batch_idx)

        loss = losses / num_batches
        error = wers / num_batches

        print(f"=> Test completed: WER = {error}, CTCloss = {loss}")

    def loader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=custom_collate)

    def load_checkpoint(self, path, map_location):
        self.model.module.load_state_dict(torch.load(path, map_location=map_location))


def main():

    parser = ArgumentParser()
    parser.add_argument('--conf', default="config.yml", help='Path to the configuration file')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.conf))

    tester = Tester(config)
    tester.test()


if __name__ == "__main__":
    main()
