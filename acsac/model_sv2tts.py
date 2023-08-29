import argparse
import os
import librosa
import numpy as np
import soundfile as sf
import cv2
from pathlib import Path
import random

import sys
sys.path.append('SV2TTS')

from encoder import inference as sv2_encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from resemblyzer import VoiceEncoder, preprocess_wav

encoder = VoiceEncoder()

encoder_path = Path('sv2_model/encoder.pt')
syn_path = Path('sv2_model/synthesizer.pt')
vocoder_path = Path('sv2_model/vocoder.pt')
sv2_encoder.load_model(encoder_path)
synthesizer = Synthesizer(syn_path)
vocoder.load_model(vocoder_path)

# text = 'This is the official implementation of protecting your voice from speech synthesize attacks.'
tgt_path = f'./examples/p287_005.wav'
out_path = './temp_wav/out.wav'
mask_path = 'temp_wav/mask.wav'
mask_infer_path = 'temp_wav/mask_infer.wav'
es_path = 'speaker_meta/p287.npy'

texts = [
    'We control complexity by establishing new languages for describing a design,'
    'each of which emphasizes particular aspects of the design and deemphasizes others.',

    'An interpreter raises the machine to the level of the user program.',

    'Everything should be made as simple as possible, and no simpler.',

    'The great dividing line between success and failurecan be expressed in five words: "I did not have time.',

    'When your enemy is making a very serious mistake,don’t be impolite and disturb him.',

    'A charlatan makes obscure what is clear; a thinker makes clear what is obscure.',

    'There are two ways of constructing a software design;one way is to make it so simple that there are obviously no deficiencies, and the other way is to make it so complicated that there are noobvious deficiencies.',

    'The three chief virtues of a programmer are: Laziness, Impatience and Hubris.',

    'All non-trivial abstractions, to some degree, are leaky.',

    'XML wasn’t designed to be edited by humans on a regular basis.'
]

def get_one_text(texts):
    return random.choice(texts)

text = get_one_text(texts)
print(len(texts))
print('the selected text is', text)

def sv2_infer(text, tgt_path, out_path):
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

sv2_infer(text, tgt_path, out_path)

def mask_param_sv2(tgt_path, text, out_path, es_path, encoder):
  print('check the original params')
  emb_spk = np.load(es_path).reshape(256,)
  emb_ori = encoder.embed_utterance(preprocess_wav(Path(tgt_path)))
  sv2_infer(text, tgt_path, out_path)
  wav_infer = preprocess_wav(Path(out_path))
  emb_infer = encoder.embed_utterance(wav_infer)
  delta_infer = np.inner(emb_spk, emb_infer)
  return emb_spk,emb_ori,emb_infer,delta_infer
for i in range(5):
    emb_spk,emb_ori,emb_infer,delta_infer = mask_param_sv2(text, tgt_path,out_path, es_path, encoder)
    print(emb_spk.shape,emb_ori.shape,emb_infer.shape,delta_infer)