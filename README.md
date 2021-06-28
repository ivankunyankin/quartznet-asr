# Documentation


This branch contains code for:

- training a QuartzNet5x5 encoder
- testing a QuartzNet5x5 encoder
- training a ConVoice decoder
- making voice conversion predictions


## Before training

1. Create and activate an environment:
```
python3 -m venv env
```
```
source env/bin/activate
```
2. Install requirements:
```
pip install -r requirements.txt
```

3. Calculate mean and std of the data before training (optional)
To calculate stats run:
```
python3 count_stats.py

positional arguments:
	conf 				(str) Path to the configuration file
	folders				(str) A list of folders containing data
```
Mean and Std will be used for data normalization

4. Adjust training parameters if the configuration file if needed

Data files should have the following structure in order to be parsed correctly
```
100_121669_000001_000000.flac
100_121669_000001_000000.transcription.txt
```

## Train a QuartzNet5x5 encoder (multiple gpus)

Run the following (requires a gpu):
```
python3 train_encoder.py

positional arguments:
  conf                  (str) Path to the configuration file
  from checkpoint		(action) Can be used to proceed training after a break
  cash					(action) Read data from memory during training
```
The training script can benifit from using multiple gpus on a single machine

## Test the trained encoder

Run the following:
```
python3 train_encoder.py

positional arguments:
  conf                  (str) Path to the configuration file
```
It will pring the resulting WER and CTC loss in the terminal as well as save intermediate logs in the logs directory

## Train a ConVoice decoder

Run the following:
```
python3 train_decoder.py

positional arguments:
  conf                  (str) Path to the configuration file
  from checkpoint		(action) Can be used to proceed training after a break
  cash					(action) Read data from memory during training
```

## Make an inference for the decoder

To make voice conversion predictions, open ```predict.py``` and specify the list of ```.flac``` files you would like to convert and the list of speakers you would like to use for conversion (ex. ["libritts_spkr_100", "libritts_spkr_101"])

After that run the script
It will save the reconstructed files in the ```out_dir``` specified in the configuration file

## Visualise training logs with Tensorboard

To visualise training logs for the encoder run the following command:
```
tensorboard --logdir logs/encoder
```
To visualise logs for the decoder run:
```
tensorboard --logdir logs/decoder
```
