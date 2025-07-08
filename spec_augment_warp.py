import random
import warnings

import torch
import torchaudio


class TimeWarp(torch.nn.Module):
    """Similar to torch.transforms.TimeStretch, but does not change the output size."""

    def __init__(self, time_warp_param, hop_length=None, n_freq=201) -> None:
        """
        Initializes the time warping.

        :param int time_warp_param: Half the window range of the "splitting axis" of the time warp.
        This axis will be in (time_warp_param, audio_length - time_warp_param).
        :param int | None hop_length: Length of hop between STFT windows
        :param int n_freq: Number of filter banks from stft.
        """
        super().__init__()

        self.time_warp_param = time_warp_param

        n_fft = (n_freq - 1) * 2
        hop_length = hop_length if hop_length is not None else n_fft // 2
        self.register_buffer("phase_advance", torch.linspace(0, torch.pi * hop_length, n_freq)[..., None])

    def forward(self, complex_specgrams) -> torch.Tensor:
        r"""
        Apply the time warping.

        In most cases the audio is correctly stretched to preserve the input size.
        However, due to rounding effect, complex_specgrams may be used for padding
        the missing values.

        :param torch.Tensor complex_specgrams: A tensor of dimension `(..., freq, num_frame)` with complex dtype.

        :return torch.Tensor: Time warped spectrogram.
        The resulting tensor is of the corresponding complex dtype
        as the input spectrogram.
        """
        if not torch.is_complex(complex_specgrams):
            warnings.warn(
                "The input to TimeWarp must be complex type. "
                "Providing non-complex tensor produces invalid results.",
                stacklevel=4,
            )
        if self.time_warp_param >= complex_specgrams.shape[2] // 2:
            raise ValueError(
                "Make sure self.time_warp_param < complex_specgrams.shape[2] // 2."
                f"Current values are {self.time_warp_param} and {complex_specgrams.shape[2] // 2}."
            )

        output_tensor = complex_specgrams.detach().clone()
        split_index = random.randint(self.time_warp_param, complex_specgrams.shape[2] - self.time_warp_param)
        w_distance = random.randint(1 - self.time_warp_param, self.time_warp_param - 1)
        warped_left = torchaudio.functional.phase_vocoder(
            complex_specgrams[:, :, :split_index],
            split_index / (split_index + w_distance),
            self.phase_advance
        )
        warped_right = torchaudio.functional.phase_vocoder(
            complex_specgrams[:, :, split_index:],
            (complex_specgrams.shape[2] - split_index) /
            (complex_specgrams.shape[2] - split_index - w_distance),
            self.phase_advance
        )
        output_tensor[:, :, :warped_left.shape[2]] = warped_left
        # Note: we may not be replacing all the values of the input tensor here
        output_tensor[:, :, warped_left.shape[2]:] = warped_right[:, :, :output_tensor.shape[2] - warped_left.shape[2]]
        return output_tensor


class SpecAugmentWarp(torch.nn.Module):
    def __init__(
        self,
        n_time_masks: int,
        time_mask_param: int,
        n_freq_masks: int,
        freq_mask_param: int,
        n_time_warps: int,
        time_warp_param: int,
    ):
        super(SpecAugmentWarp, self).__init__()
        if n_time_warps < 0 or n_time_masks < 0 or n_freq_masks < 0:
            raise ValueError(
                "n_time_warps, n_time_masks, n_freq_masks should all be >=0."
                f"Currently at {n_time_warps}, {n_time_masks} and {n_freq_masks}."
            )
        self.n_time_warps = n_time_warps
        self.time_warp = TimeWarp(time_warp_param)
        self.n_time_masks = n_time_masks
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.frequency_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)

    def forward(self, complex_specgrams):
        for _ in range(self.n_time_warps):
            complex_specgrams = self.time_warp(complex_specgrams)

        if self.n_time_masks + self.n_freq_masks == 0:
            return complex_specgrams

        power_spectrum = complex_specgrams.abs()
        for _ in range(self.n_time_masks):
            power_spectrum = self.time_mask(power_spectrum)
        for _ in range(self.n_freq_masks):
            power_spectrum = self.frequency_mask(power_spectrum)
        masked_areas = (power_spectrum - complex_specgrams).abs() < 1e-8

        # Note: masked areas in complex plan are replaced by norm of the complex value
        return complex_specgrams * masked_areas + power_spectrum * ~masked_areas
