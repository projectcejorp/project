
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
from torch import Tensor

import copy
import numpy as np
import os
import pickle
import soundfile as sf
import torch

from data_utils import plot_mel, mel_load, file2mel, mel2wav, griffin_lim
from defense_utils import SpectrogramMask, is_overlap, win2block, mask_param, generate_mask_params
from defense import mask_search
from inference import  chous_infer
from parameters import args

if __name__ == "__main__":
    # load Es()
    encoder = VoiceEncoder()

    # example paths
    wav_path = './examples/p287_{args.sid}.wav'
    src_path = './examples/content.wav'
    out_path = './temp_wav/infer_raw.wav'
    mask_path = 'temp_wav/mask.wav'
    mask_infer_path = 'temp_wav/mask_infer.wav'
    es_path = 'speaker_meta/p287.npy'

    # get the init params for optimization
    emb_spk, emb_ori, emb_infer, delta_infer = mask_param(wav_path, src_path, out_path, es_path, encoder)

    print(f'check the embedding shape {emb_ori.shape}')

    sp_mask, xd = mask_search(args,
                              wav_path,
                              src_path,
                              out_path,
                              mask_path,
                              mask_infer_path,
                              es_path,
                              encoder)

    print(f'Check saved defense stratefy pairs: {sp_mask}')

    wav_xd = mel2wav(xd.T,
                     args.sample_rate,
                     args.preemph,
                     args.n_fft,
                     args.hop_length,
                     args.win_length,
                     args.n_mels,
                     args.ref_db,
                     args.max_db,
                     args.top_db)


    sf.write(mask_path, wav_xd, args.sample_rate)
    print(f'saving the modified speech...')

    xd_mel = mel_load(mask_path)
    # we here use Chou's model for inference as an example
    # The defense framework support any black-box speech synthesis models,
    # .e.g, X_infer(src_path, xd_mel.T, mask_infer_path, other_params)
    chous_infer(src_path, xd_mel.T, mask_infer_path)
    print(f'Saving the synthetic speech based on modified speech...')

    # an example for speaker mask
    # spk_mask = [(13, 'Zero'), (14, 'Zero'), (15, 'Zero'), (5, 'Zero'), (6, 'Zero'), (11, 'AN'), (12, 'AN')]
    # xd = ite_spkmask(wav_path, sp_mask, args)

