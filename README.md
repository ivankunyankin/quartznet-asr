## QuartzNet

Lightweight PyTorch implementation of QuartzNet (https://arxiv.org/pdf/1910.10261.pdf). <!-- You can choose between three different version of the model: ```5x5, 10x5, 15x5```. For details refer to the article. -->

<p align="center"><img width="250" src="https://developer-blogs.nvidia.com/wp-content/uploads/2019/12/QuartzNet-architecture.png"></a></p>

<div align="center"><i><small>QuartzNet BxR architecture</small></i></div>

### Features

1. Allows to choose between three different model sizes: ![5x5](https://img.shields.io/badge/-5x5-blue), ![10x5](https://img.shields.io/badge/-10x5-blue), ![15x5](https://img.shields.io/badge/-15x5-blue). For details refer to the article.  

2. Easily customisable  

3. Allows training using a cpu, single or multiple ![gpu](https://img.shields.io/badge/-gpus-green)  

4. Suitable for training in ![colab](https://img.shields.io/badge/-Google%20Colab-orange) and ![aws](https://img.shields.io/badge/-AWS-orange) spot instances as it allows to continue training after a break-down.

### Table of contents

1. Installation :books:
2. Default training :books:
3. Train custom data :books:
4. Hyperparameters :books:
5. Augmentation
6. Things that are different from the article 

## How to install

1. Clone the repository
``` 
git clone https://github.com/ivankunyankin/quartznet-asr.git
cd quartznet-asr 
```

2. Create and activate an environment 
``` 
python3 -m venv env 
source env/bin/activate 
```

3. Install the dependencies 
``` 
pip3 install -r requirements.txt 
```

## How to use

[(back to the top)](#quartznet)

### Training

This guide shows training using LibriTTS dataset. However, the code can be easily adjusted to be trained with different data. More details [here](docs/data.md).

Also, hyperparameters for training described [here](docs/hparams.md).

1. Download the data from [here](https://openslr.org/60/) and unzip into ```LibriTTS``` folder.

2. Run the following to start training:

```
python3 train.py
```
Add ```--cache``` parameter to read the data into memory for faster training.  
Add ```--from_checkpoint``` parameter to continue training from a checkpoint.

The code can benifit from using single or multiple gpus on a single machine

The code allows to continue training from a checkpoint that is especially conveniet when training using Google Colab or AWS spot instances. More details here

### Testing

Run the following:
```
python3 test.py
```

With default settings specified in the ```config.yaml``` the code will test the trained model on test-clean part of LibriTTS

It will print the resulting WER (word error rate) and CTC loss values in the terminal as well as save intermediate logs in the logs directory

5. Visualising training logs with Tensorboard

To visualise training logs run the following command:
```
tensorboard --logdir logs
```

## Contribution

[(back to the top)](#quartznet)

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change

## Acknowledgements

[(back to the top)](#quartznet)

I found inspiration for TextTransform class greedy decoder in [this](https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch) post.
