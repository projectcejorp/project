# Protecting Your Voice From Speech Synthesis Attacks.

This repository hosts the codes of our project, which focuses on protecting a speaker's voice from speech synthesis attacks. In this project, we propose two defense schemes that can be used by the speaker to process his or her speeches before publishing them on social media platforms or sending them to others. The processed speeches cannot only significantly degrade the performance of speech synthesis systems but also keep the sound of the speaker's voice so that they can still be used for normal purposes.

# What does this repository contain?

- Data processing of audio files

- How we modify speech samples with our proposed three modification methods in the frequency domain:
  - Zero Mask
  - AN-Mask
  - GB-Mask

- How we iteratively find the optimal frequency-modification pairs.

- How we protect a single speech (SampleMask).

- How we protect arbitrary speeches of a speaker (SpeakerMask).

- Examples of running a defense.

# Setup of the project.

Since many audio-based libraries require specific versions, we recommend setting up a new environment to avoid "dependencies" issues:

Step 1: Create a new conda environment with Python version 3.8

```bash
conda create --name my_env python=3.8
conda activate my_env
```

Step 2: Install requirements

```bash
pip install -r requirements.txt
```

Step 3: Manually install Resemblyzer without its dependencies (to avoid conflicts with requirements.txt)
```bash
pip install Resemblyzer==0.1.3 --no-deps
```

Step 4: Download experimental data 

You can download the data from [here]([https://drive.google.com/file/d/1vVOakChiRJwV8MCXc_2HjHjRcXxzuIEW/view?usp=sharing](https://drive.google.com/file/d/1Nz8FshcndOdttxp873h7xHonWo8WZoYA/view?usp=sharing]). After downloading, please copy the data to the root directory. A detailed description can be found in the submitted pdf file.

# Examples of running defense.

To perform the defense on a target speech sample:

```bash
python example.py --sid 005 --tau 0.06 
```

To perform the defense on multiple speech samples of a speaker and get SpeakerMask:
```bash
python speakermask.py --spk_id p287 --tau 0.06 --system chou 
```

Evaluation:

The Attack results will be printed and saved with the above commands, for example:

```bash
Check the quality of synthetic speech before and after the modification: before defense: 0.743, after defense: 0.536
Attack against Resembylzer before and after the defense: success, after: fail
The ASR of the synthetic speeches based on speaker p287 is 10% after the defense.
```

# Integration Tips

Our defense schemes support any black-box speech synthesis systems. If you would like to try more speech synthesis systems, we recommend using general functions to represent VC models and TTS models so that they can easily combine with our defense schemes. We also encourage later works to explore more potential modification methods in the frequency domain.

# Reference
For pretrain the speech synthesis systems, you can refer to the original papers  of [Chou’s](https://arxiv.org/abs/1904.05742), [AutoVC](https://arxiv.org/abs/1905.05879), and [SV2TTS](https://arxiv.org/abs/1806.04558). For the Attack-VC (the baseline in our paper), please refer to their [paper](https://arxiv.org/abs/2005.08781). In this repository, we provide some audio examples for running the codes. The full dataset we used in the paper, CSTR VCTK, is public and can be found [here](https://datashare.ed.ac.uk/handle/10283/3443).


Parts of our codebase were inspired by or adapted from the following repositories:

- [**attack-vc**](https://github.com/cyhuang-tw/attack-vc) 
- [**AutoVC**](https://github.com/cyhuang-tw/AutoVC) 
- [**Real-Time-Voice-Cloning**](https://github.com/CorentinJ/Real-Time-Voice-Cloning)

We thank the authors of these repositories for making their code available to the community.

