#### 3. (optional) Calculate mean and std of the data before training 

To calculate a single value for mean and std, run:
```
python3 count_stats.py --folders LibriTTS
```
To calculate a separate mean and std for each mel channel, run:
```
python3 count_stats.py --folders LibriTTS --per_channel
```
The script will walk over folders specified in ```--folders```, calculate stats and save them into a ```.npy``` file.

Stats are used for data normalization during training






The code can be used to train a model on LibriTTS dataset which has the following structure:

Audio file are in ```.wav``` or ```.flac``` formats
Transcripts have the same name as corresponding audio files and end with ```.normalized.txt```

If you want to train the model on different data, you can either transform the data or adjust the LibriDataset class in dataset.py

You can find more details about different training parameters here.