import torch
import torchaudio
import matplotlib as matplotlib
import matplotlib.cm


def concat(x, y, exp_to=2, cat_on=1):

    assert len(x.shape) == len(y.shape)

    dims = [i if idx == exp_to else -1 for idx, i in enumerate(x.shape)]
    y = y.expand(*dims)
    return torch.cat([x, y], dim=cat_on)


def save_spec(spec):

    cm = matplotlib.cm.get_cmap('gray')

    normed = (spec - spec.min()) / (spec.max() - spec.min())
    mapped = cm(normed)

    return torch.from_numpy(mapped).flip(0).permute(2, 0, 1)


def augment(spec, piece_length=30, freq_mask_param=10, time_mask_param=6):

    num = spec.shape[1] // piece_length

    freq = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
    time = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

    if num > 1:
        pieces = []

        for i in range(1, num + 1):

            start = piece_length * (i - 1)
            end = piece_length * i
            piece = spec[:, start:end]

            if i == num:
                piece = spec[:, start:]

            freq(piece)
            time(piece)

            pieces.append(piece)

        return torch.cat(pieces, dim=1)

    else:
        freq(spec)
        time(spec)

        return spec


def custom_collate(data):
    """
       data: is a list of tuples with (melspec, transcript, input_length, label_length, speaker_id), where:
        - 'melspec' is a tensor of arbitrary shape
        - 'transcript' is an encoded transcript - list of integers
        - input_length - is length of the spectrogram - represents time
        - label_length - is length of the encoded label
        - speaker_id - is a scalar - represents a unique speaker id for the speaker encoder
    """
    melspecs, texts, input_lengths, label_lengths, speakers = zip(*data)

    max_inp_len = max(input_lengths)
    max_label_len = max(label_lengths)

    n_mels = melspecs[0].shape[0]
    features = torch.zeros((len(data), n_mels, max_inp_len))
    labels = torch.zeros((len(data), max_label_len))

    for i in range(len(data)):
        input_length = data[i][0].size(1)
        label_length = data[i][3]
        features[i] = torch.cat([data[i][0], torch.zeros((n_mels, max_inp_len - input_length))], dim=1)
        labels[i] = torch.cat([data[i][1], torch.zeros((max_label_len - label_length))])

    return features, labels, torch.tensor(input_lengths), torch.tensor(label_lengths), speakers


CHAR_MAP = """
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


class TextTransform:

    """Maps characters to integers and vice versa"""
    def __init__(self, char_map_str):

        self.char_map_str = char_map_str
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
