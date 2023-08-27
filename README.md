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


## Defense

To perform the defense toward a target speech sample, you could follow the example command below:

```bash
python main.py --sid 005 --tau 0.06 --eps 0.1 --ksize 11 --std 1.5 --b_num 16
