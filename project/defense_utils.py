from matplotlib import pyplot as plt
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.signal import lfilter
from torch import Tensor
from typing import Dict, Optional, Tuple

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
import tqdm
import yaml
import torchaudio

from data_utils import plot_mel, mel_load, file2mel, mel2wav, griffin_lim
from model_chous import AdaInVC
from inference import chous_infer
from parameters import args
import numpy as np
import copy


class SpectrogramMask:
    '''
    -spec: FxT spectrogram
    -b_num: frequency block number
    -step_b: mask range 32 for speakerMask, 64 for sampleMask
    -spk_m: sample or speaker mask
    '''
    def __init__(self, spec: np.ndarray, b_num: int, step_b=0, spk_m=False):
        self.spec = copy.deepcopy(spec)
        self.b_num = b_num
        self.F = self.spec.shape[0]  # Number of frequency bands (e.g., 512)
        self.mask_range = self.F // self.b_num
        self.step_b = step_b
        self.spk_m = spk_m

    def _calculate_f0_f1(self, b_idx: int) -> tuple:
        '''
        define mask range
        '''
        step = self.mask_range if self.spk_m else 2 * self.mask_range
        # print('check step', step)

        if b_idx >= self.b_num:
            raise ValueError('Block index out of range')

        f0 = b_idx * self.mask_range
        f1 = f0 + step if self.step_b == 0 else f0 + self.step_b
        return f0, f1

    def zero_mask(self, b_idx: int) -> np.ndarray:
        '''
        return: modified xd with Zero Mask
        '''
        f0, f1 = self._calculate_f0_f1(b_idx)
        self.spec[f0:f1, :] = 0
        return self.spec

    def an_mask(self, b_idx: int, distribution: str = 'G', eps: float = None) -> np.ndarray:
        '''
        return: modified xd with AN-Mask
        '''
        f0, f1 = self._calculate_f0_f1(b_idx)
        if distribution == 'G':
            noise = np.random.normal(0, 1, self.spec[f0:f1, :].shape)
        elif distribution == 'L':
            noise = np.random.laplace(0, 1, self.spec[f0:f1, :].shape)
        else:
            raise ValueError(f"Unsupported distribution type: {distribution}")

        noise = np.clip(noise, -eps, eps)
        self.spec[f0:f1, :] += noise
        return self.spec

    def gb_mask(self, b_idx: int, ksize: int, std: float) -> np.ndarray:
        '''
        return: modified xd with AN-Mask
        '''
        f0, f1 = self._calculate_f0_f1(b_idx)
        blur = cv2.GaussianBlur(self.spec[f0:f1, :], (ksize, ksize), std)
        self.spec[f0:f1, :] = blur
        return self.spec

    def apply_mask(self, idx: int, mask_type: str, **kwargs) -> np.ndarray:
        if mask_type == 'Zero':
            return self.zero_mask(idx)
        elif mask_type == 'AN':
            return self.an_mask(idx, kwargs.get('distribution', 'G'), kwargs.get('eps'))
        elif mask_type == 'GB':
            return self.gb_mask(idx, kwargs.get('ksize'), kwargs.get('std'))
        else:
            raise ValueError(f"Unsupported mask type: {mask_type}")


def win2block(m_win):
    '''
    Convert window mask to block masks
    -m_win: the pairs of window mask
    return the pairs of block mask
    '''
    sp_mask = []

    for win, mask in m_win:
        if win != 15:  # If it's not the last window, break it into two blocks.
            sp_mask.append((win, mask))
            sp_mask.append((win + 1, mask))
        else:  # If it's the last window, keep it as is.
            sp_mask.append((win, mask))

    return sp_mask


def is_overlap(idx, selected_indices):
    '''Check if window idx overlaps with any index in selected_indices.'''
    if (idx - 1 in selected_indices) or (idx + 1 in selected_indices):
        return True

    else:
        return False


def mask_param(wav_path, src_path, out_path, es_path, encoder):
      '''
      -wav_path: path of the target voice
      -src_path: path of the source voice
      -out_path: the path of the synthetic speech'

      Return:
      The vanilla params for optimization
      '''
      emb_spk = np.load(es_path).reshape(256,)
      emb_ori = encoder.embed_utterance(preprocess_wav(Path(wav_path)))
      wav_mel = mel_load(wav_path)
      chous_infer(src_path,wav_mel.T,out_path)
      wav_infer = preprocess_wav(Path(out_path))
      emb_infer = encoder.embed_utterance(wav_infer)
      delta_infer = np.inner(emb_ori, emb_infer)

      return emb_spk, emb_ori, emb_infer,delta_infer

def generate_mask_params(i, mask, args):
    params = {
        'idx': i,
        'mask_type': mask,
    }

    if mask == 'AN':
        params.update({
            'distribution': args.distribution,
            'eps': args.eps
        })
    elif mask == 'GB':
        params.update({
            'ksize': args.ksize,
            'std': args.std
        })

    return params

def compute_delta_Qd(mask_path, base_mel, i, mask, args):
    params = generate_mask_params(i, mask, args)
    spec_masker = SpectrogramMask(base_mel, args.b_num,0,True)
    xd = spec_masker.apply_mask(**params)

    # Convert mel spectrogram back to waveform
    wav_xd = mel2wav(xd.T,
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

    sf.write(mask_path, wav_xd, args.sample_rate)
    wav = preprocess_wav(Path(mask_path))
    emb = encoder.embed_utterance(wav)
    delta_Qd = 1 - np.inner(emb_ori, emb)

    return xd, delta_Qd

def ite_spkmask(wav_path, sp_mask, args):
    '''
    apply speaker mask to any speeches
    '''
    base_mel = mel_load(wav_path)
    last_win, last_mask = None, None
    mel_list = []

    for win, mask in sp_mask:
        print(f'win {win}, mask: {mask}')
        xd, delta_Qd = compute_delta_Qd(mask_path, base_mel, win, mask, args)

        base_mel = xd
        mel_list.append(xd)

        print('check delta Qd', delta_Qd)

        if delta_Qd > args.tau_d:
            last_win, last_mask = win, mask
            break

    low, high = 0, 32
    b_range = (low + high) // 2
    base_mel = mel_list[-2]

    for _ in range(100):
        params = generate_mask_params(last_win, last_mask, args)
        spec_masker = SpectrogramMask(base_mel, args.b_num, b_range, True)
        xd = spec_masker.apply_mask(**params)

        wav_xd = mel2wav(xd.T,
                         args.
                         sample_rate,
                         args.preemph,
                         args.n_fft,
                         args.hop_length,
                         args.win_length,
                         args.n_mels,
                         args.ref_db,
                         args.max_db,
                         args.top_db
                         )

        sf.write(mask_path, wav_xd, args.sample_rate)
        wav = preprocess_wav(Path(mask_path))
        emb = encoder.embed_utterance(wav)
        delta_Qd = 1 - np.inner(emb_ori, emb)

        if args.tau_d - args.rou < delta_Qd < args.tau_d + args.rou:
            break
        elif delta_Qd < args.tau_d - args.rou:
            low = b_range
        elif delta_Qd > args.tau_d + args.rou:
            high = b_range
        elif high - low == 1:  # This checks for the "last split" condition
            print("Reached the last split.")
            break

        b_range = (low + high) // 2

    return xd

