# Protecting Your Voice From Speech Synthesis Attacks.

This repository hosts the codes of our project, “Protecting Your Voice From Speech Synthesis Attacks.” We modify speeches in the frequency domains to avoid one’s voice getting cloned by speech synthesis systems, preventing improper usage in Voicer conversion (VC)  and Text-to-speech (TTS).

# What does this repository contain?

- Data processing of audio files

- How we modify speech samples with proposed three modification methods in the frequency domain:
  - Zero Mask
  - AN-Mask
  - GB-Mask

- How we iteratively find the optimum frequency-modification pairs.

- How we protect a single speech (SampleMask).

- How we protect arbitrary speeches of a speaker (SpeakerMask).

- Examples of running a defense.

# Setup of the project.

Since many audio-based libraries require specific versions, we recommend setting up the environments by running:

```bash
example bash
```


## Example of running a defense.

To perform the defense toward a target speech sample, you could follow the example command below:

```bash
python main.py --sid 005 --tau 0.06 --eps 0.1 --ksize 11 --std 1.5 --b_num 16
```

To perform the defense toward arbitary speech samples of a speaker, you could follow the example command below:
```bash
python main.py --sid 005 --tau 0.06 --eps 0.1 --ksize 11 --std 1.5 --b_num 16
```


# Others
Our defense scheme supports any black-box speech synthesis systems. If you would like to try more speech synthesis systems,, we strongly recommend use general functions to represent VC models and TTS models, such as

vc_infer(src_path, tgt_path,out_path, **model_config)
tts_infer(text, tgt_path,out_path,**model_config)

So that it can easily combine with our defense scheme, Also, we encourage later works to explore more potential modification methods in the frequency domains.

# Reference

For pretrain the speech synthesis systems, you can refer to the [Chou’s](https://arxiv.org/abs/1904.05742), [AutoVC]([URL2](https://arxiv.org/abs/1905.05879)), and [SV2TTS]([URL3](https://arxiv.org/abs/1806.04558)) original papers.
For the [Attack-VC](the baseline in our paper), please refer to the their [paper](https://arxiv.org/abs/2005.08781): 


Parts of our codebase were inspired by or adapted from the following repositories:

- [**attack-vc**](https://github.com/cyhuang-tw/attack-vc) 
- [**AutoVC**](https://github.com/cyhuang-tw/AutoVC) 
- [**Real-Time-Voice-Cloning**](https://github.com/CorentinJ/Real-Time-Voice-Cloning)

We extend our gratitude to the authors of these repositories for making their code available to the community.




