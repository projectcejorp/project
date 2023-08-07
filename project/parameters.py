import argparse

'''
Audio params
'''
parser = argparse.ArgumentParser(description='Parameters for audio processing')
parser.add_argument('--sample_rate', type=int, default=24000, help='Sample rate.')
parser.add_argument('--preemph', type=float, default=0.97, help='Preemphasis.')
parser.add_argument('--n_fft', type=int, default=2048, help='Number of FFTs.')
parser.add_argument('--hop_length', type=int, default=300, help='Hop length.')
parser.add_argument('--win_length', type=int, default=1200, help='Window length.')
parser.add_argument('--n_mels', type=int, default=512, help='Number of Mel filters.')
parser.add_argument('--ref_db', type=float, default=20.0, help='Reference dB.')
parser.add_argument('--max_db', type=float, default=100.0, help='Max dB.')
parser.add_argument('--top_db', type=float, default=15.0, help='Top dB.')
parser.add_argument('--n_iter', type=int, default=100, help='Number of iterations for griffin_lim.')

'''
Defense params
'''
parser.add_argument('--ksize', type=int, default=11, help='kernel size for GB-Mask.')
parser.add_argument('--std', type=float, default=1.5, help='standard deviation for GB-Mask.')
parser.add_argument('--eps', type=float, default=0.1, help='eps for noise clipping.')
parser.add_argument('--spk_m', type=bool, default=False, help='Whether apply speaker mask.')
parser.add_argument('--distribution', type=str, default='G', help='Distribution of the noise.')
parser.add_argument('--b_num', type=int, default='16', help='Block num for the spectrogram.')
parser.add_argument('--tau_d', type=float, default='0.06', help='threshold for sample distribution.')
parser.add_argument('--rou', type=float, default='0.02', help='relax bounds.')
parser.add_argument('--sid', type=str, default='005', help='sample id.')

args = parser.parse_args()