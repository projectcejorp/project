import copy
import librosa
import numpy as np
import os
import pickle
import soundfile as sf
import torch
import torch.nn as nn
import yaml

from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.signal import lfilter
from typing import Dict, Optional, Tuple
from model_chous import AdaInVC
from parameters import args

def inv_mel_matrix(sample_rate: int, n_fft: int, n_mels: int) -> np.array:
    m = librosa.filters.mel(sample_rate, n_fft, n_mels)
    p = np.matmul(m, m.T)
    d = [1.0 / x if np.abs(x) > 1e-8 else x for x in np.sum(p, axis=0)]

    return np.matmul(m.T, np.diag(d))


def normalize(mel: np.array, attr: Dict) -> np.array:
    mean, std = attr["mean"], attr["std"]
    mel = (mel - mean) / std

    return mel


def denormalize(mel: np.array, attr: Dict) -> np.array:
    mean, std = attr["mean"], attr["std"]
    mel = mel * std + mean

    return mel


def file2mel(
    audio_path: str,
    sample_rate: int,
    preemph: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    ref_db: float,
    max_db: float,
    top_db: float
) -> np.array:
    wav, _ = librosa.load(audio_path, sr=sample_rate)
    wav = np.append(wav[0], wav[1:] - preemph * wav[:-1])
    linear = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(linear)
    mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels)
    mel = np.dot(mel_basis, mag)

    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mel = mel.T.astype(np.float32)

    return mel


def mel2wav(
    mel: np.array,
    sample_rate: int,
    preemph: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    ref_db: float,
    max_db: float,
    top_db: float,
) -> np.array:
    mel = mel.T
    mel = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db
    mel = np.power(10.0, mel * 0.05)

    inv_mat = inv_mel_matrix(sample_rate, n_fft, n_mels)
    mag = np.dot(inv_mat, mel)
    wav = griffin_lim(mag, hop_length, win_length, n_fft)
    wav = lfilter([1], [1, -preemph], wav)

    return wav.astype(np.float32)


def griffin_lim(
    spect: np.array,
    hop_length: int,
    win_length: int,
    n_fft: int,
    n_iter: Optional[int] = 100,
) -> np.array:
    X_best = copy.deepcopy(spect)
    for _ in range(n_iter):
        X_t = librosa.istft(X_best, hop_length, win_length, window="hann")
        est = librosa.stft(X_t, n_fft, hop_length, win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spect * phase
    X_t = librosa.istft(X_best, hop_length, win_length, window="hann")
    y = np.real(X_t)

    return y


def load_model(model_dir: str) -> Tuple[nn.Module, Dict, Dict, str]: # load Chou's model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attr_path = os.path.join(model_dir, "attr.pkl")
    cfg_path = os.path.join(model_dir, "config.yaml")
    model_path = os.path.join(model_dir, "model.ckpt")

    attr = pickle.load(open(attr_path, "rb"))
    config = yaml.safe_load(open(cfg_path, "r"))
    model = AdaInVC(config["model"]).to(device)
    model.load_state_dict(torch.load(model_path))

    return model, config, attr, device


def mel_load(voice_path):
    mel = file2mel(voice_path,
                   args.sample_rate,
                   args.preemph,
                   args.n_fft,
                   args.hop_length,
                   args.win_length,
                   args.n_mels,
                   args.ref_db,
                   args.max_db,
                   args.top_db)

    mel = mel.T

    return mel


def plot_mel(mel):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(mel, x_axis="s", y_axis="mel", sr= args.sample_rate, hop_length=  rgs.hop_length)
    plt.colorbar(format="%+2.f")
