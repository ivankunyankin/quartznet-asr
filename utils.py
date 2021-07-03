import torch
import librosa
import torchaudio
import numpy as np
import matplotlib as matplotlib
import matplotlib.cm
from models import QuartzNet
from torch.nn.utils.rnn import pad_sequence


def create_model(model, in_channels, out_channels):

    models = ["quartznet5x5", "quartznet10x5", "quartznet15x5"]
    assert model in models, f"Unknown model name. Expected one of {models}, but got {model}"

    if model == "quartznet5x5":
        return QuartzNet(repeat=1, in_channels=in_channels, out_channels=out_channels)

    elif model == "quartznet10x5":
        return QuartzNet(repeat=2, in_channels=in_channels, out_channels=out_channels)

    elif model == "quartznet15x5":
        return QuartzNet(repeat=3, in_channels=in_channels, out_channels=out_channels)


def audio_to_mel(x, hparams):

    spec = librosa.feature.melspectrogram(
        x,
        sr=hparams["sr"],
        n_fft=hparams["n_fft"],
        win_length=hparams["win_length"],
        hop_length=hparams["hop_length"],
        power=1,
        fmin=0,
        fmax=8000,
        n_mels=hparams["n_mels"]
    )

    spec = np.log(np.clip(spec, a_min=1e-5, a_max=None))
    spec = torch.FloatTensor(spec)

    return spec


def save_spec(spec):

    cm = matplotlib.cm.get_cmap('gray')

    normed = (spec - spec.min()) / (spec.max() - spec.min())
    mapped = cm(normed)

    return torch.from_numpy(mapped).flip(0).permute(2, 0, 1)


def augment(spec, chunk_size=30, freq_mask_param=10, time_mask_param=6):

    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=int(freq_mask_param), iid_masks=True)
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=int(time_mask_param), iid_masks=True)

    num_chunks = spec.shape[1] // int(chunk_size)

    if num_chunks <= 1:
        freq_mask(spec)
        time_mask(spec)
        return spec
    else:
        chunks = torch.split(spec, chunk_size, dim=1)
        to_be_masked = torch.stack(list(chunks[:-1]), dim=0).unsqueeze(1)
        time_mask(to_be_masked)
        freq_mask(to_be_masked)
        masked = to_be_masked.squeeze(1).permute(1, 0, 2).reshape((spec.shape[0], -1))
        return torch.cat([masked, chunks[-1]], dim=1)


def custom_collate(data):

    """
   data: is a list of tuples with (melspec, transcript, input_length, label_length), where:
    - 'melspec' is a tensor of arbitrary shape
    - 'transcript' is an encoded transcript - list of integers
    - input_length - is length of the spectrogram - represents time - int
    - label_length - is length of the encoded label - int
    """

    melspecs, texts, input_lengths, label_lengths = zip(*data)

    specs = [torch.transpose(spec, 0, 1) for spec in melspecs]
    specs = pad_sequence(specs, batch_first=True)
    specs = torch.transpose(specs, 1, 2)

    labels = pad_sequence(texts, batch_first=True)

    return specs, labels, torch.tensor(input_lengths), torch.tensor(label_lengths)


class TextTransform:

    """Maps characters to integers and vice versa"""
    def __init__(self):

        self.char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19    
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """

        self.char_map = {}
        self.index_map = {}

        for line in self.char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []

        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)

        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []

        for i in labels:
            string.append(self.index_map[i])

        return ''.join(string)

    def decode(self, output, labels, label_lengths, blank_label=28, collapse_repeated=True):

        arg_maxes = torch.argmax(output, dim=2)

        decodes = []
        targets = []

        for i, args in enumerate(arg_maxes):  # for each sample in the batch
            decode = []
            targets.append(self.int_to_text(labels[i][:label_lengths[i]].tolist()))

            for j, index in enumerate(args):  # for each predicted character in the sample
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j - 1]:
                        continue
                    decode.append(index.item())

            decodes.append(self.int_to_text(decode))

        return decodes, targets
