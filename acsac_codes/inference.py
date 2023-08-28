import copy
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import yaml
from matplotlib import pyplot as plt
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.signal import lfilter
from torch import Tensor
from data_utils import file2mel, mel2wav, mel_load, normalize, denormalize
from model_autovc import get_embed, pad_seq
# from model_chous import *
from parameters import args

sys.path.append('SV2TTS')
from encoder import inference as sv2_encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

# def chous_infer(src_path, tgt_mel, out_path:str):
def chous_infer(src_path, tgt_path, out_path, model, attr):

    src_mel = file2mel(src_path,
                       args.sample_rate,
                       args.preemph,
                       args.n_fft,
                       args.hop_length,
                       args.win_length,
                       args.n_mels,
                       args.ref_db,
                       args.max_db,
                       args.top_db)

    tgt_mel = mel_load(tgt_path).T
    src_mel = normalize(src_mel, attr)
    tgt_mel = normalize(tgt_mel, attr)
    src_mel = torch.from_numpy(src_mel).T.unsqueeze(0).to('cuda')
    tgt_mel = torch.from_numpy(tgt_mel).T.unsqueeze(0).to('cuda')

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



def autovc_infer(src_path, tgt_path, out_path, model, vocoder, wav2mel):

    tgt_mel, tgt_sr = torchaudio.load(tgt_path)
    tgt_mel = wav2mel(tgt_mel, tgt_sr).to('cuda')
    tgt_emb = get_embed(model.speaker_encoder, tgt_mel)
    src, src_sr = torchaudio.load(src_path)
    src_mel = wav2mel(src, src_sr).to('cuda')
    src_emb = get_embed(model.speaker_encoder, src_mel)
    src_mel, len_pad = pad_seq(src_mel)
    src_mel = src_mel[None, :]

    with torch.no_grad():
      _, mel, _ = model(src_mel, src_emb, tgt_emb)
      mel = mel[0, :, :] if len_pad == 0 else mel[0, :-len_pad, :]
      wav = vocoder.generate([mel])[0].data.cpu().numpy()

    sf.write(out_path, wav.astype(np.float32), wav2mel.sample_rate)


def sv2_infer(text, tgt_path, out_path, sv2_encoder, synthesizer, vocoder):

    pre_wav = sv2_encoder.preprocess_wav(tgt_path)
    tgt_emb = sv2_encoder.embed_utterance(pre_wav)
    texts = [text]
    embeds = [tgt_emb]
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    generated_wav = sv2_encoder.preprocess_wav(generated_wav)

    sf.write(out_path, generated_wav.astype(np.float32), synthesizer.sample_rate)