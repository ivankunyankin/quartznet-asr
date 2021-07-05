## QuartzNet

Lightweight PyTorch implementation of QuartzNet (https://arxiv.org/pdf/1910.10261.pdf). <!-- You can choose between three different version of the model: ```5x5, 10x5, 15x5```. For details refer to the article. -->

<p align="center"><img width="250" src="https://developer-blogs.nvidia.com/wp-content/uploads/2019/12/QuartzNet-architecture.png"></a></p>

<div align="center"><i><small>QuartzNet BxR architecture</small></i></div>

### Features

1. Allows to choose between three different model sizes: ![5x5](https://img.shields.io/badge/-5x5-blue), ![10x5](https://img.shields.io/badge/-10x5-blue), ![15x5](https://img.shields.io/badge/-15x5-blue). For details refer to the article.  
2. Easily customisable  
3. Allows training using a cpu, single or multiple ![gpu](https://img.shields.io/badge/-gpus-green).  
4. Suitable for training in ![colab](https://img.shields.io/badge/-Google%20Colab-orange) and ![aws](https://img.shields.io/badge/-AWS-orange) spot instances as it allows to continue training after a break-down.  

### Table of contents

1. [Installation](#installation)  
2. [Default training](#default-training)  
3. [Train custom data](https://github.com/ivankunyankin/quartznet-asr/issues/1) :books:
4. [Hyperparameters](https://github.com/ivankunyankin/quartznet-asr/issues/2) :books:
5. [Things that are different compared to the article](https://github.com/ivankunyankin/quartznet-asr/issues/3)

## Installation

1. Clone the repository
``` 
git clone https://github.com/ivankunyankin/quartznet-asr.git
cd quartznet-asr 
```

2. Create an environment  and install the dependencies
``` 
python3 -m venv env 
source env/bin/activate 
pip3 install -r requirements.txt 
```

## Default training

[(back to the top)](#quartznet)

### Training

This guide shows training QuartzNet5x5 model using LibriTTS dataset.

1. Download the data from [here](https://openslr.org/60/) and unzip into ```LibriTTS``` folder.

2. Run the following to start training:

```
python3 train.py
```
Add ```--cache``` parameter to read the data into memory for faster training.  
Add ```--from_checkpoint``` parameter to continue training from a checkpoint.

### Testing

```
python3 test.py
```

The code will test the trained model on test-clean subset of LibriTTS.  
It will print the resulting WER (word error rate) and CTC loss values as well as save intermediate logs in the logs directory

### Tensorboard

```
tensorboard --logdir logs
```

## Contribution

[(back to the top)](#quartznet)

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change

## Acknowledgements

[(back to the top)](#quartznet)

I found inspiration for TextTransform class and Greedy decoder in [this](https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch) post.
