
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
from defense_utils import SpectrogramMask, is_overlap, win2block, mask_param, generate_mask_params


def mask_search(args,
                wav_path: str,
                src_path: str,
                out_path: str,
                mask_path: str,
                mask_infer_path:str,
                es_path:str,
                encoder):

    sp_mask = None
    tgt_mel = mel_load(wav_path)
    base_mel = copy.deepcopy(tgt_mel)
    emb_spk, emb_ori, emb_infer, delta_infer = mask_param(wav_path, src_path, out_path, es_path, encoder)

    best_wi = []
    win_mask = []
    delta_Qd_best, delta_QI_best = 0, 0

    # Iterate over block numbers
    for epoch in range(args.b_num):
        fsen_list = []

        for i in range(args.b_num):
            print('\n')

            if is_overlap(i, best_wi) or (i in best_wi):
                print(f"Skip frequency, Window idx: {i}")
                continue

            for mask in ['Zero', 'AN', 'GB']:

                params = generate_mask_params(i, mask, args)
                spec_masker = SpectrogramMask(tgt_mel if epoch == 0 else base_mel, args.b_num, 0, False)
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

                # Embed utterance
                wav = preprocess_wav(Path(mask_path))
                emb = encoder.embed_utterance(wav)

                # Calculate quality change of raw speech
                delta_Qd = 1 - np.inner(emb_ori, emb)

                # Mask inference
                out_infer = mask_infer_path
                chous_infer(src_path, xd.T, out_infer)
                wav_infer = preprocess_wav(Path(out_infer))
                emb_infer = encoder.embed_utterance(wav_infer)

                # Calculate quality change of synthetic speeches
                delta_QI = delta_infer - np.inner(emb_ori, emb_infer)

                # Calculate frequency sensitivity
                QI_change = delta_QI - delta_QI_best
                Qd_change = delta_Qd - delta_Qd_best

                if delta_Qd > args.tau_d + args.rou:
                    print(f'large sample distortion.')
                    fsen = -100
                elif  QI_change < 0 :
                    print(f'Inverse defense effects .')
                    fsen = -100
                else:
                    fsen =  QI_change / Qd_change

                print(f"Window idx: {i}, Mask type: {mask}, QI_change: { QI_change:.4f}, Qd_change: {Qd_change:.4f}, freq_sensi: {fsen:.4f}")

                fsen_list.append((i, mask, fsen, delta_Qd, delta_QI))

        fsen_list.sort(key=lambda x: x[2], reverse=True)
        win_i = fsen_list[0][0]           # window index
        mask_i = fsen_list[0][1]          # mask type
        fs = fsen_list[0][2]              # frequency sensitivity in last iteration
        delta_Qd_best = fsen_list[0][3]   # delta_Qd in last iteration
        delta_QI_best = fsen_list[0][4]   # delta_QI in last iteration

        if fs > 0:
            best_wi.append(win_i)
            win_mask.append((win_i, mask_i))
        else:
            print('no avaliable mask, search stops')

        print(f'Epoch: {epoch}, best_wi: {best_wi}, pairs: {win_mask}')

        best_params = generate_mask_params(win_i, mask_i, args)
        spec_masker = SpectrogramMask(tgt_mel if epoch == 0 else base_mel, args.b_num, 0, False)
        base_mel = spec_masker.apply_mask(** best_params)

        if delta_Qd_best > args.tau_d:
            print(f'Search stops due to constraint, the current Delta_Qd is {delta_Qd_best:.3f}')
            break

    sp_mask = win2block(win_mask)

    return sp_mask, base_mel