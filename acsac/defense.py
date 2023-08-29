import sys
import numpy as np
import os
import copy
import pickle
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
from data_utils import mel_load, mel2wav
from model_chous import AdaInVC
from model_autovc import Wav2Mel
from inference import chous_infer, autovc_infer, sv2_infer
from defense_utils import SpectrogramMask, is_overlap, win2block, generate_mask_params
from parameters import args

sys.path.append('SV2TTS')
from encoder import inference as sv2_encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from resemblyzer import VoiceEncoder, preprocess_wav

def mask_search_test(
    args,
    tgt_path: str,         # speech of target speaker
    src_path: None,        # source voice is only needed for VC model
    text: None,            # text is only needed for TTS model
    out_path: str,
    mask_path: str,
    mask_infer_path: str,
    es_path: str,
    encoder,
    model_config: dict = None
) -> np.array:

    sample_mask = None
    tgt_mel = mel_load(tgt_path)         # convert waveform to mel spectrogram
    base_mel = copy.deepcopy(tgt_mel)    # the masked spectrogram in current iteration

    emb_ori = encoder.embed_utterance(preprocess_wav(Path(tgt_path)))
    emb_spk = emb_ori if args.load_es == False else np.load(es_path).reshape(256, )

    QI_list = []

    for r in range(args.repeat):
        if args.system == 'chou':
            chous_infer(src_path, tgt_path, out_path, **model_config)
        elif args.system == 'autovc':
            autovc_infer(src_path, tgt_path, out_path, **model_config)
        elif args.system == 'sv2tts':
            sv2_infer(text, tgt_path, out_path, **model_config)

        wav_infer = preprocess_wav(Path(out_path))
        emb_infer = encoder.embed_utterance(wav_infer)
        QI = np.inner(emb_spk, emb_infer)
        QI_list.append(QI)

    QI = np.mean(QI_list)    # get the quality of the raw synthetic speech (we aim to decrease this value with our defense scheme)

    best_wi = []
    win_mask = []
    delta_Qd_best, delta_QI_best = 0, 0
    win_num = args.b_num if not args.system == 'autovc' else args.b_num-2 # AutoVC automatically trim high frequency thus do not have to search the top 2 window

    for epoch in range(win_num): # iterate over all frequency windows
        fsen_list = []

        for i in reversed(range(win_num)):

            if is_overlap(i, best_wi) or (i in best_wi):  # check if the window is searched
                continue

            print("****************************************************************************")
            print("Window idx  |  Mask type  |  delta_QI_diff  |  delta_Qd_diff  |  freq_sensi")
            print("****************************************************************************")

            for mask in ['Zero', 'AN', 'GB']:   #iterate over three modification methods
                params = generate_mask_params(i, mask, args) # get the hyperparameter of Modification methods
                spec_masker = SpectrogramMask(tgt_mel if epoch == 0 else base_mel, args.b_num, 0, False)
                xd = spec_masker.apply_mask(**params) # get the modified spectrogram

                # convert spectrogram to waveform
                wav_xd = mel2wav(xd.T, args.sample_rate, args.preemph, args.n_fft, args.hop_length,
                                 args.win_length, args.n_mels, args.ref_db, args.max_db, args.top_db)

                sf.write(mask_path, wav_xd, args.sample_rate) # get the current protected speech
                wav = preprocess_wav(Path(mask_path))
                emb = encoder.embed_utterance(wav)

                Qd = np.inner(emb_ori, emb)
                delta_Qd = 1 - Qd    # quality change of raw speech

                delta_QI_list = []

                for r in range(args.repeat): #Since Autovc and SV2TTS have some randomness when synthesizing, it is recommended to set repeat = 5.
                    if args.system == 'chou':
                        chous_infer(src_path, mask_path, mask_infer_path, **model_config)
                    elif args.system == 'autovc':
                        autovc_infer(src_path, mask_path, mask_infer_path, **model_config)
                    elif args.system == 'sv2tts':
                        sv2_infer(text, mask_path, mask_infer_path, **model_config)
                        print('\n')

                    wav_infer = preprocess_wav(Path(mask_infer_path))
                    emb_infer = encoder.embed_utterance(wav_infer)
                    delta_QI = QI - np.inner(emb_spk, emb_infer)
                    delta_QI_list.append(delta_QI)

                delta_QI = np.mean(delta_QI_list)
                delta_QI_diff = delta_QI - delta_QI_best  #quality change (QI) difference with last iteration
                delta_Qd_diff = delta_Qd - delta_Qd_best  #quality change (Qd) difference with last iteration

                if delta_Qd > args.tau_d + args.rou: # quality of defense sample is below constraint
                    fsen = -100
                elif delta_QI_diff < 0: # the synthetic speech can not continue to degrade compared with last round
                    fsen = -100
                else:
                    # Calculate frequency sensitivity
                    fsen = delta_QI_diff / delta_Qd_diff

                print(f"{i:<12} |  {mask:<10} |  {delta_QI_diff:.4f}     |     {delta_Qd_diff:.4f}      |     {fsen:.4f}")

                fsen_list.append((i, mask, fsen, delta_Qd, delta_QI))

            print("****************************************************************************")

        fsen_list.sort(key=lambda x: x[2], reverse=True) # sort the frequency-method pairs by frequency sensitivity
        win_i, mask_i, fs, delta_Qd_best, delta_QI_best = fsen_list[0]  # get meta information

        if fs > 0:
            best_wi.append(win_i)
            win_mask.append((win_i, mask_i))
        else:
            print(f'No available mask, search stops.')
            break

        print(f'Epoch: {epoch}, best_wi: {best_wi}, pairs: {win_mask}')

        best_params = generate_mask_params(win_i, mask_i, args)
        spec_masker = SpectrogramMask(tgt_mel if epoch == 0 else base_mel, args.b_num, 0, False)
        base_mel = spec_masker.apply_mask(**best_params) # save the base spectrogram for next iteration

    sample_mask = win2block(win_mask)  # convert frequency window-method pair in to frequency block pair

    return sample_mask, base_mel
