import numpy as np
import soundfile as sf
import torch

from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
from data_utils import load_model, mel2wav
from defense_utils_v2 import emb_simi
from defense_v2 import mask_search_test
from inference_v2 import chous_infer
from parameters import args


def main():
    encoder = VoiceEncoder()  # Load Es()

    # Example paths
    tgt_path = f'./examples/p287_{args.sid}.wav'          # speech of target speaker
    src_path = f'./examples/content.wav'                  # speech of source speaker
    infer_path = f'./temp_wav/raw_infer_{args.sid}.wav'   # path of raw synthetic speech
    es_path = f'speaker_meta/p287.npy'                    # speaker embedding (optional)

    # Path definitions
    mask_path = 'temp_wav/mask.wav'                       # temporary path of modified speech
    mask_infer_path = 'temp_wav/mask_infer.wav'           # temporary path of synthetic speech based on modified speech
    xd_path = f'./examples/xd_{args.sid}.wav'             # Final modified speech (xd in paper)
    xd_infer_path = f'./examples/xd_infer_{args.sid}.wav' # Final synthetic speech based on modified speech (W(xd) in paper)

    # Load model and set model configurations
    model, config, attr, device = load_model("choumodel")
    model_config = {
        'model': model,
        'attr': attr
    }

    # synthesize raw speech and check its quality
    print(f'Synthesizing raw speech ...')
    chous_infer(src_path, tgt_path, infer_path, **model_config)
    QI_before = emb_simi(tgt_path, infer_path, encoder)
    print(f'Check the quality of raw synthetic speech: {QI_before:.3f}')

    # Get the best frequency-modification pairs and modified speech
    sp_mask, xd = mask_search_test(
        args, tgt_path, src_path, None, infer_path, mask_path,
        mask_infer_path, es_path, encoder, model_config
    )

    # Save xd to waveform
    wav_xd = mel2wav(
        xd.T, args.sample_rate, args.preemph, args.n_fft, args.hop_length,
        args.win_length, args.n_mels, args.ref_db, args.max_db, args.top_db
    )
    sf.write(xd_path, wav_xd, args.sample_rate)

    # synthesize modified speech and check its quality
    print(f'Synthesizing modified speech ...')
    chous_infer(src_path, xd_path, xd_infer_path, **model_config)
    QI_after = emb_simi(tgt_path, xd_infer_path, encoder)
    print(f'Check the quality of modified synthetic speech: {QI_after:.3f}')


if __name__ == "__main__":
    main()



    # an example for speaker mask
    # spk_mask = [(13, 'Zero'), (14, 'Zero'), (15, 'Zero'), (5, 'Zero'), (6, 'Zero'), (11, 'AN'), (12, 'AN')]
    # xd = ite_spkmask(wav_path, sp_mask, args)

