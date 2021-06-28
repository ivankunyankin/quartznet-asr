import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import librosa


hparams = {
    "sr": 16000,
    "n_mels": 80,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 256,
}


def dynamic_range_compression(x):
    return torch.log(torch.clamp(x, 1e-5))


def normalize_spect(x):
    return (x + 5.0) / 5.0


def audio_to_spect(audio, hparams, apply_normalize_spect=True, apply_padding=True, enforce_int_hopsizes=True, device=torch.device('cpu')):

    if isinstance(audio, np.ndarray):

        return audio_to_spect(torch.FloatTensor(audio).to(device), hparams=hparams).detach().cpu().numpy()

    elif isinstance(audio, torch.Tensor):

        if len(audio.shape) == 1:
            return audio_to_spect(audio.unsqueeze(0), hparams=hparams).squeeze(0)

        elif len(audio.shape) == 2:

            lin_to_mel_mat = torch.FloatTensor(librosa.filters.mel(sr=hparams["sr"], n_fft=hparams["win_length"], n_mels=hparams["n_mels"])).to(device)

            if enforce_int_hopsizes:
                n_too_much = audio.size()[-1] % hparams["hop_length"]
                if n_too_much > 0:
                    audio = audio[..., :-n_too_much]

            if apply_padding:
                orig_len = audio.size()[-1]
                n_to_pad = (hparams["win_length"] - hparams["hop_length"]) // 2
                audio = F.pad(audio.unsqueeze(0), (n_to_pad, n_to_pad), mode='reflect').squeeze(0)
            cmplx = torch.stft(
                audio,
                win_length=hparams["win_length"],
                hop_length=hparams["hop_length"],
                n_fft=hparams["win_length"],
                center=False,
                window=torch.hann_window(hparams["win_length"]).to(device),
                return_complex=True
            )
            spect = torch.abs(cmplx)
            mel = lin_to_mel_mat @ spect
            mel = dynamic_range_compression(mel)

            if apply_normalize_spect:
                mel = normalize_spect(mel)

            if enforce_int_hopsizes and apply_padding:
                assert orig_len == mel.size()[-1] * hparams["hop_length"], "audio_to_spect with padding should give exactly 1 column per hop_length samples"

        elif len(audio.shape) == 3:
            return audio_to_spect(audio.squeeze(1), hparams=hparams)

        else:
            raise Exception("WhisppSTFT obtained audio tensor of unexpected shape")

        return mel

    else:
        raise Exception("WhisppSTFT obtained audio of unexpected type")

