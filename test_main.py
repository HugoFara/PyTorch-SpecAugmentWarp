import matplotlib.pyplot as plt
import torchaudio

import spec_augment_warp as saw

my_audio = "MLKDream.wav"
audio_tensor, frequency = torchaudio.load(my_audio, normalize=True)

spec_augment_warp_trans = saw.SpecAugmentWarp(
    n_time_warps=2,
    time_warp_param=50,
    n_time_masks=2,
    time_mask_param=180,
    n_freq_masks=2,
    freq_mask_param=60
)
spectrogram = torchaudio.transforms.Spectrogram(power=None)(audio_tensor)
fig, axes = plt.subplots(2, 1)
axes[0].matshow(spectrogram.squeeze().abs().flip(0))
spectrogram = spec_augment_warp_trans(spectrogram)
axes[1].matshow(spectrogram.squeeze().abs().flip(0))
plt.show()

# Convert back to an audio (if necessary)
audio_tensor = torchaudio.transforms.InverseSpectrogram()(spectrogram)

# You can also use the TimeWarp transform
time_warp_transform = saw.TimeWarp(time_warp_param=20)
spectrogram = time_warp_transform(spectrogram)
