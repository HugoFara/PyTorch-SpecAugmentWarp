# PyTorch-SpecAugmentWarp

A pure PyTorch implementation of [SpecAugment](https://arxiv.org/abs/1904.08779) that adds Time Warping,
and preserves phase information.
As of July 2025, torchaudio implements SpecAugment partially, drops phase,
and do not provide any documentation about their implementation.

![SpecAugment Illustration](SpecAugment%20Illustrated.png)

This version adds the missing time warping feature.
See the [long explanation](#specaugment-rabbit-hole)

## Install

Just copy the content of ``spec_augment_warp.py``.
You will only need PyTorch and torchaudio.

To get all the requirements:

```sh
pip install torch torchaudio
```

## Usage

SpecAugmentWarp is implemented as PyTorch module.

```python
import torchaudio
import spec_augment_warp as saw

audio_tensor, frequency = torchaudio.load("audio.wav")

spec_augment_warp_trans = saw.SpecAugmentWarp(
    n_time_warps=2,
    time_warp_param=60,
    n_time_masks=2,
    time_mask_param=80,
    n_freq_masks=2,
    freq_mask_param=40
)
# Get the complex spectrogram (important for phase conservation)
spectrogram = torchaudio.transforms.Spectrogram(power=None)(audio_tensor)
# Apply SpecAugment 
spectrogram = spec_augment_warp_trans(spectrogram)

# Convert back to an audio (if necessary)
audio_tensor = torchaudio.transforms.InverseSpectrogram()(spectrogram)
```

Additionally, we also provide the TimeWarp function.
It is similar to TimeStretch, but preserve the input shape by applying a counterbalanced
speed-up and a slow-down around a pivot axis.

```python
import torchaudio
import spec_augment_warp as saw

audio_tensor, frequency = torchaudio.load("audio.wav")

# You can also use the TimeWarp transform directly
time_warp_transform = saw.TimeWarp(time_warp_param=60)
spectrogram = torchaudio.transforms.Spectrogram(power=None)(audio_tensor)
spectrogram = time_warp_transform(spectrogram)
```

## SpecAugment Rabbit Hole

SpecAugment is not fully implemented on PyTorch.
The reason is it is difficult as there is no equivalent to the matrix transform time warp.

However, in 2023, [@xiaohui-zhang](https://github.com/xiaohui-zhang) proposed an implementation in 
[audio/pull/#3309](https://github.com/pytorch/audio/pull/3309) that has been merged,
and released in [torchaudio 2.1.0](https://github.com/pytorch/audio/releases/tag/v2.1.0)
He did not give much any information on why time warp was missing.

Regarding the torchaudio documentation, it is not better. 
As of the [2.7.0 documentation](https://docs.pytorch.org/audio/2.7.0/tutorials/audio_feature_augmentation_tutorial.html#specaugment),
they only give the following information.

> [SpecAugment](https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html) is a popular spectrogram augmentation technique.
> 
> torchaudio implements torchaudio.transforms.TimeStretch(), torchaudio.transforms.TimeMasking() and torchaudio.transforms.FrequencyMasking().

And go on to this usage of each of those functions, without even linking to their torchaudio.transforms.SpecAugment class.
In fact, you will not find any documentation on this class!

> Time warp and phase are missing, why do we care?

Time warp increases performances by 1% (so not much).
Missing phase information can totally garble the audio.
It happens when you have several frequencies close to each other. 

## Other implementations

Some shoutout to people that did their own versions:

- [DemisEom](https://github.com/DemisEom/SpecAugment): PyTorch + Tensorflow. He used the Tensorflow function for the time warp.
- [zcaceres/spec_augment](https://github.com/zcaceres/spec_augment): PyTorch + Tensorflow. He used the Tensorflow function for the time warp.

Hugging Face: they did a good job applying a SpecAugment on the output of the feature encoder 
(see [this from line for the WavLM version](https://github.com/huggingface/transformers/blob/896e9cea1ade521b2648f4798218550f6c72190c/src/transformers/models/wavlm/modeling_wavlm.py#L1020)).

However, they are also missing time warps.
