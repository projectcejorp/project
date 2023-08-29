import numpy as np
import os
import pickle
import soundfile as sf
import sys
import torch

from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
from data_utils import load_model, mel2wav
from model_autovc import Wav2Mel
from defense_utils import emb_simi, get_one_text, evaluate_resemblyzer_sample
from defense import mask_search_test
from inference import chous_infer, autovc_infer, sv2_infer
from parameters import args

sys.path.append('SV2TTS')
from encoder import inference as sv2_encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as sv2_vocoder


def main():
    encoder = VoiceEncoder()  # load Es()
    attack_count = 0

    # load model and set model configurations
    if args.system == 'chou':
        model, config, attr, device = load_model("choumodel")
        model_config = {
            'model': model,
            'attr': attr
        }

    elif args.system == 'autovc':
        autovc_model = torch.jit.load('auto_vc_model/model.pt').to(args.device)
        autovc_vocoder = torch.jit.load('auto_vc_model/vocoder.pt').to(args.device)
        wav2mel = Wav2Mel()

        model_config = {
            'model': autovc_model,
            'vocoder': autovc_vocoder,
            'wav2mel': wav2mel
        }

    elif args.system == 'sv2tts':
        encoder_path = Path('sv2_model/encoder.pt')
        syn_path = Path('sv2_model/synthesizer.pt')
        vocoder_path = Path('sv2_model/vocoder.pt')

        sv2_encoder.load_model(encoder_path)
        sv2_synthesizer = Synthesizer(syn_path)
        sv2_vocoder.load_model(vocoder_path)

        model_config = {
            'sv2_encoder': sv2_encoder,
            'synthesizer': sv2_synthesizer,
            'vocoder': sv2_vocoder
        }
    else:
        raise ValueError(f"Unsupported system value: {args.system}. Expected 'chou', 'autovc', or 'sv2tts'.")

    print(f'Targeting defending against {args.system} model.')

    tgt_root = os.path.join('target_speaker', args.spk_id) # the speeches of the target speaker
    src_root = os.path.join('source_speaker', args.spk_id) # the speeches of the source speaker

    # initialze the speaker_mask (will add the searched pairs of 10 speeches into the list)
    speaker_mask = []
    es_path = f'speaker_meta/{args.spk_id}.npy'

    # results_xd: modified speeches of all speakers
    # results_xd_infer: synthetic speeches based on modified speeches
    # temp_wav: the root for storing temporary files

    directories = ['results_xd', 'results_xd_infer', 'temp_wav']
    for dir_name in directories:
        os.makedirs(os.path.join(dir_name, args.spk_id), exist_ok=True)

    xd_root = os.path.join('results_xd', args.spk_id)
    xd_infer_root = os.path.join('results_xd_infer', args.spk_id)

    infer_path = os.path.join('temp_wav', args.spk_id, 'infer_raw.wav')       # raw synthetic speech
    mask_path = os.path.join('temp_wav', args.spk_id, 'mask.wav')             # temporary xd
    mask_infer_path = os.path.join('temp_wav', args.spk_id, 'mask_infer.wav') # temporary xd_infer

    tgt_files = sorted(os.listdir(tgt_root))
    src_files = []
    if args.system in ['chou', 'autovc']:
        src_files = sorted(os.listdir(src_root))

    try:
        for idx, tgt_wav in enumerate(tgt_files):
            tgt_path = os.path.join(tgt_root, tgt_wav)
            emb_ori = encoder.embed_utterance(preprocess_wav(Path(tgt_path)))
            print(f'The current path of target speaker: {tgt_path}\n')

            if args.system in ['chou', 'autovc']:
                src_wav = src_files[idx]
                src_path = os.path.join(src_root, src_wav)
                sample_mask, xd = mask_search_test(args, tgt_path, src_path, None, infer_path,      #  text=None of VC model
                                                   mask_path, mask_infer_path, es_path, encoder, model_config)

            if args.system == 'sv2tts':
                text = get_one_text()
                print(f'Text to be converted to speech: {text}')
                sample_mask, xd = mask_search_test(args, tgt_path, None, text, infer_path,     #source=None for TTS
                                                   mask_path, mask_infer_path, es_path, encoder, model_config)

            print('Check the sample mask', sample_mask)
            speaker_mask.append(sample_mask)

            wav_xd = mel2wav(xd.T, args.sample_rate, args.preemph, args.n_fft, args.hop_length,
                             args.win_length, args.n_mels, args.ref_db, args.max_db, args.top_db)

            xd_path = os.path.join(xd_root, f'{tgt_wav[:8]}_xd.wav')
            xd_infer_path = os.path.join(xd_infer_root, f'{tgt_wav[:8]}_with_{src_wav[:8]}_.wav')

            sf.write(xd_path, wav_xd, args.sample_rate)
            print(f'saving the modified speech...')

            # Synthesize speech based on the modification
            if args.system == 'chou':
                chous_infer(src_path, xd_path, xd_infer_path, **model_config)
            elif args.system == 'autovc':
                autovc_infer(src_path, xd_path, xd_infer_path, **model_config)
            elif args.system == 'sv2tts':
                sv2_infer(text, xd_path, xd_infer_path, **model_config)

            print(f'Saving the synthetic speech based on modified speech...')

            QI_before = emb_simi(tgt_path, infer_path, encoder)
            QI_after = emb_simi(tgt_path, xd_infer_path, encoder)
            # compare the quality of synthetic speeches before and after the defense scheme
            print(f'Check the quality of synthetic speech: before and after the modification: {QI_before:.3f}, after: {QI_after:.3f}')

            status_before = "success" if evaluate_resemblyzer_sample(args, QI_before) == 1 else "fail"
            status_after = "success" if evaluate_resemblyzer_sample(args, QI_after) == 1 else "fail"

            print(f'Attack against Resembylzer before and after the defense: {status_before}, after: {status_after}')

            if status_after == "success":
                attack_count += 1

    except FileNotFoundError:
        print(f"File not found. Check the paths provided.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print(f'The searched speaker_mask is {speaker_mask}.')
    print(f'The ASR of the target speaker after the defense is: {100*attack_count/10}%')


if __name__ == "__main__":
    main()