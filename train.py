import os
import yaml
import torch
import random
import datetime
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from jiwer import wer
import torch.optim as optim
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from IPython.display import clear_output
from torch.utils.data import DataLoader

from dataset import LibriDataset
from utils import TextTransform, save_spec, custom_collate, create_model

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class Trainer:

    def __init__(self, config, rank, world_size, from_checkpoint):

        self.device = rank
        self.world_size = world_size

        # Parameters
        self.batch_size = config["batch_size"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.start_epoch = 1
        self.epochs = config["epochs"] + 1
        self.use_onecyclelr = config["use_onecyclelr"]

        # Data
        self.train_set = LibriDataset(config, "train")
        self.val_set = LibriDataset(config, "val")
        self.train_loader = self.loader(self.train_set)
        self.val_loader = self.loader(self.val_set)
        self.processor = TextTransform()

        # Model
        self.model = create_model(
            model=config["model"],
            in_channels=config["spec_params"]["n_mels"],
            out_channels=len(self.processor.char_map) + 1  # for blank token
        )
        self.model.to(self.device)

        if self.world_size:
            self.model = DistributedDataParallel(self.model, device_ids=[self.device])

        self.criterion = nn.CTCLoss(blank=len(self.processor.char_map))
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))

        if self.use_onecyclelr:
            self.scheduler = self.oneCycleLR(config)

        if from_checkpoint:
            if os.path.exists(os.path.join(self.checkpoint_dir, "model_last.pt")):
                if self.world_size:
                    map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
                    self.load_checkpoint(self.checkpoint_dir, map_location)
                    print(f"=> Rank {self.device}. Loaded checkpoint")
                else:
                    self.load_checkpoint(self.checkpoint_dir, map_location=self.device)
                    print("=> Loaded checkpoint")

                with open(os.path.join(self.checkpoint_dir, "last_epoch.txt"), "r") as f:
                    last_epoch = int(f.read())
                    last_batch_idx = last_epoch * len(self.train_loader) - 1
                    self.start_epoch = last_epoch + 1
                    if self.use_onecyclelr:
                        self.scheduler = self.oneCycleLR(config, last_epoch=last_batch_idx)
            else:
                print("* Checkpoint not found")

        if not self.device == "cpu":
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging
        if self.device == 0 or not self.world_size:

            now = datetime.datetime.now()
            path = os.path.join(config["log_dir"], now.strftime("%Y:%m:%d_%H:%M:%S"))
            self.checkpoint_path = os.path.join(self.checkpoint_dir, now.strftime("%Y:%m:%d_%H:%M:%S"))
            self.last_epoch_path = os.path.join(self.checkpoint_dir, "last_epoch.txt")

            self.train_writer = SummaryWriter(os.path.join(path, "train"))
            self.val_writer = SummaryWriter(os.path.join(path, "val"))

            with open(f"{path}/hparams.yml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)

    def train(self):

        best_loss = None

        # Training
        for epoch in range(self.start_epoch, self.epochs):

            self.train_step(epoch)
            if self.world_size:
                dist.barrier()

            loss = self.val_step(epoch)
            if self.world_size:
                dist.barrier()
                print(f'Finished epoch {epoch}, rank {self.device}/{self.world_size}')

            if self.device == 0 or not self.world_size:

                self.save_checkpoint(self.checkpoint_dir, postfix="last")
                print("=> Checkpoint updated")

                if best_loss is None:
                    best_loss = loss

                elif loss < best_loss:
                    self.save_checkpoint(self.checkpoint_dir, postfix="best")
                    best_loss = loss

                if epoch == self.epochs - 1:
                    self.move_checkpoints()
                    if os.path.exists(self.last_epoch_path):
                        os.remove(self.last_epoch_path)
                else:
                    with open(self.last_epoch_path, "w") as f:
                        f.write(str(epoch))

            if self.world_size:
                dist.barrier()

    def train_step(self, step):

        self.model.train()
        loop = tqdm(self.train_loader)
        losses = 0
        num_batches = 0

        for batch_idx, (specs, transcripts, input_lengths, label_length) in enumerate(loop):

            clear_output(wait=True)
            loop.set_description(f"Device: {self.device}. Epoch {step} (train)")
            self.optimizer.zero_grad()

            specs = specs.to(self.device)
            transcripts = transcripts.to(self.device)
            input_lengths = input_lengths.to(self.device)
            label_length = label_length.to(self.device)

            if not self.device == "cpu":
                with torch.cuda.amp.autocast():
                    output = self.model(specs)
                    output = output.permute(2, 0, 1)
                    output = F.log_softmax(output, dim=2)
                    loss = self.criterion(output, transcripts, input_lengths, label_length)
                    losses += loss
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                output = self.model(specs)
                output = output.permute(2, 0, 1)
                output = F.log_softmax(output, dim=2)
                loss = self.criterion(output, transcripts, input_lengths, label_length)
                losses += loss
                loss.backward()
                self.optimizer.step()

            if self.use_onecyclelr:
                self.scheduler.step()

            loop.set_postfix(loss=loss.item())
            num_batches += 1

            if self.device == 0 or not self.world_size:

                self.train_writer.add_scalar(f"Epoch {step}: loss", loss, global_step=batch_idx)

                for param_group in self.optimizer.param_groups:
                    rate = param_group["lr"]

                self.train_writer.add_scalar("Learning Rate", rate, global_step=batch_idx + len(self.train_loader) * (step - 1))

                if batch_idx % 100 == 0:
                    rand_idx = random.randint(0, specs.shape[0] - 1)
                    self.train_writer.add_image(f"Epoch {step} (train): augmented specs", save_spec(specs[rand_idx].to("cpu").detach()), global_step=batch_idx)

        if self.device == 0 or not self.world_size:
            loss = losses / num_batches
            self.train_writer.add_scalar("CTC loss", loss, global_step=step)

    def val_step(self, step):

        self.model.eval()
        loop = tqdm(self.val_loader)
        losses = 0
        wers = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (specs, transcripts, input_lengths, label_length) in enumerate(loop):

                clear_output(wait=True)
                loop.set_description(f"Device: {self.device}. Epoch {step} (val)")

                specs = specs.to(self.device)
                transcripts = transcripts.to(self.device)
                input_lengths = input_lengths.to(self.device)
                label_length = label_length.to(self.device)

                if not self.device == "cpu":
                    with torch.cuda.amp.autocast():
                        output = self.model(specs)
                        output = output.permute(2, 0, 1)
                        output = F.log_softmax(output, dim=2)
                        loss = self.criterion(output, transcripts, input_lengths, label_length)
                else:
                    output = self.model(specs)
                    output = output.permute(2, 0, 1)
                    output = F.log_softmax(output, dim=2)
                    loss = self.criterion(output, transcripts, input_lengths, label_length)

                losses += loss

                loop.set_postfix(loss=loss.item())

                num_batches += 1

                if self.device == 0 or not self.world_size:

                    decoded_preds, decoded_targets = self.processor.decode(output.permute(1, 0, 2), transcripts, label_length)
                    error = wer(decoded_targets, decoded_preds)
                    wers += error

                    # Save training logs to Tensorboard
                    rand_idx = random.randint(0, specs.shape[0] - 1)
                    self.val_writer.add_text(f"Epoch {step} (val): preds", decoded_preds[rand_idx], global_step=batch_idx)
                    self.val_writer.add_text(f"Epoch {step} (val): targets", decoded_targets[rand_idx], global_step=batch_idx)
                    self.val_writer.add_scalar(f"Epoch {step}: loss", loss, global_step=batch_idx)

        loss = losses / num_batches
        error = wers / num_batches

        if self.device == 0 or not self.world_size:
            self.val_writer.add_scalar("CTC loss", loss, global_step=step)
            self.val_writer.add_scalar("WER", error, global_step=step)

        return loss

    def oneCycleLR(self, hparams, last_epoch=-1):

        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=float(hparams["max_lr"]),
            steps_per_epoch=len(self.train_loader),
            epochs=int(hparams["epochs"]),
            div_factor=float(hparams["div_factor"]),
            pct_start=float(hparams["pct_start"]),
            last_epoch=last_epoch
        )

        return scheduler

    def loader(self, dataset):

        if self.world_size:
            sampler = DistributedSampler(dataset, rank=self.device, num_replicas=self.world_size)
            loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, collate_fn=custom_collate)
        else:
            loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=custom_collate)

        return loader

    def save_checkpoint(self, path, postfix=""):

        if not os.path.exists(path):
            os.mkdir(path)

        if self.world_size:
            torch.save(self.model.module.state_dict(), os.path.join(path, f"model_{postfix}.pt"))
        else:
            torch.save(self.model.state_dict(), os.path.join(path, f"model_{postfix}.pt"))

        torch.save(self.optimizer.state_dict(), os.path.join(path, f"optimizer_{postfix}.pt"))

    def load_checkpoint(self, path, map_location):

        if self.world_size:
            self.model.module.load_state_dict(torch.load(os.path.join(path, "model_last.pt"), map_location=map_location))
        else:
            self.model.load_state_dict(torch.load(os.path.join(path, "model_last.pt"), map_location=map_location))

        self.optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer_last.pt"), map_location=map_location))

    def move_checkpoints(self):

        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        os.rename(os.path.join(self.checkpoint_dir, "model_last.pt"), os.path.join(self.checkpoint_path, "model_last.pt"))
        os.rename(os.path.join(self.checkpoint_dir, "model_best.pt"), os.path.join(self.checkpoint_path, "model_best.pt"))
        os.rename(os.path.join(self.checkpoint_dir, "optimizer_last.pt"), os.path.join(self.checkpoint_path, "optimizer_last.pt"))
        os.rename(os.path.join(self.checkpoint_dir, "optimizer_best.pt"), os.path.join(self.checkpoint_path, "optimizer_best.pt"))


def init_process(rank, size, backend="nccl"):

    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)


def train_dist(rank, world_size, config, from_checkpoint):

    init_process(rank, world_size)
    print(f"Rank {rank}/{world_size} training process initialized.\n")

    trainer = Trainer(config, rank, world_size, from_checkpoint)
    dist.barrier()
    print(f"Rank {rank}/{world_size} initialised trainer.\n")

    trainer.train()


def main():

    parser = ArgumentParser()
    parser.add_argument('--conf', default="config.yml", help='Path to the configuration file')
    parser.add_argument('--from_checkpoint', action="store_true", help='Continue training from the last checkpoint')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.conf))
    from_checkpoint = args.from_checkpoint

    world_size = torch.cuda.device_count()

    if world_size > 1:
        mp.spawn(train_dist,
                 args=(world_size, config, from_checkpoint),
                 nprocs=world_size,
                 join=True)

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = Trainer(config, rank=device, world_size=None, from_checkpoint=from_checkpoint)
        print("=> Initialised trainer")
        print("=> Training...")
        trainer.train()


if __name__ == "__main__":
    main()
