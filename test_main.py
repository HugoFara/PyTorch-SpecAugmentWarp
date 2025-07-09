import matplotlib.pyplot as plt
import torchaudio
import torch

import spec_augment_warp as saw


def display_spectrogram(spectrogram, augmented_spectrogram):
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].set_title("Original spectrogram")
    axes[0].matshow(spectrogram.squeeze().abs())
    axes[0].invert_yaxis()
    axes[0].get_yaxis().set_visible(False)
    axes[1].set_title("Application of SpecAugment")
    axes[1].matshow(augmented_spectrogram.squeeze().abs())
    axes[1].invert_yaxis()
    axes[1].get_yaxis().set_visible(False)
    axes[1].set_xlabel("Time (s)")
    plt.show()


my_audio = "audio.wav"
audio_tensor, frequency = torchaudio.load(my_audio, normalize=True)

spec_augment_warp_trans = saw.SpecAugmentWarp(
    n_time_warps=2,
    time_warp_param=80,
    n_time_masks=2,
    time_mask_param=80,
    n_freq_masks=3,
    freq_mask_param=60
)
spectrogram = torchaudio.transforms.Spectrogram(power=None)(audio_tensor)
spectrogram += torch.rand_like(spectrogram)  # Add some noise to increase contrast
augmented_spectrogram = spec_augment_warp_trans(spectrogram)
display_spectrogram(spectrogram, augmented_spectrogram)

# Convert back to an audio (if necessary)
audio_tensor = torchaudio.transforms.InverseSpectrogram()(augmented_spectrogram)

# You can also use the TimeWarp transform
time_warp_transform = saw.TimeWarp(time_warp_param=20)
spectrogram = time_warp_transform(spectrogram)
