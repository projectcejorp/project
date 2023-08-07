## Acknowledgements
## Parts of the code in this repository are based on the work by Attack-VC. You can find the original source code: https://github.com/cyhuang-tw/attack-vc

import copy
import cv2
import librosa
import librosa.display
import numpy as np
import os
import pickle
import random
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import yaml
from matplotlib import pyplot as plt
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.signal import lfilter
from torch import Tensor
from typing import Dict, Optional, Tuple

from model_chous import *
from data_utils import *
from parameters import args

def chous_infer(src_path, tgt_mel, out_path: str):
    '''
    src_path: the path of the source voice
    tgt_mel: the mel spectrogram of the target voice
    out_path: the path of the synthetic speech
    '''
    model, config, attr, device = load_model("./vcmodel")

    src_mel = file2mel(src_path,
                       args.sample_rate,
                       args.preemph,
                       args.n_fft,
                       args.hop_length,
                       args.win_length,
                       args.n_mels,
                       args.ref_db,
                       args.max_db,
                       args.top_db

                       )
    src_mel = normalize(src_mel, attr)
    tgt_mel = normalize(tgt_mel, attr)
    src_mel = torch.from_numpy(src_mel).T.unsqueeze(0).to(device)
    tgt_mel = torch.from_numpy(tgt_mel).T.unsqueeze(0).to(device)

    with torch.no_grad():
        out_mel = model.inference(src_mel, tgt_mel)
        out_mel = out_mel.squeeze(0).T

    out_mel = denormalize(out_mel.data.cpu().numpy(), attr)
    out_wav = mel2wav(out_mel,
                      args.sample_rate,
                      args.preemph,
                      args.n_fft,
                      args.hop_length,
                      args.win_length,
                      args.n_mels,
                      args.ref_db,
                      args.max_db,
                      args.top_db
                      )

    sf.write(out_path, out_wav, args.sample_rate)