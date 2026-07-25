import math

import torch

import comfy.k_diffusion.sampling as k_diffusion_sampling
import comfy.samplers


VIDEO_CHANNELS = 128
AUDIO_CHANNELS = 8
AUDIO_FREQUENCY_BINS = 16


class LTXSetImageSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 1728, "min": 64, "max": 8192, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 1152, "min": 64, "max": 8192, "step": 32},
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "size"
    CATEGORY = "LTX/support"

    def size(self, width, height):
        return (width, height)


def latent_video_shape(width, height, length):
    if width % 32 or height % 32:
        raise ValueError("LTX loop width and height must be divisible by 32")
    if length < 9:
        raise ValueError("LTX loop length must be at least 9 frames")
    return VIDEO_CHANNELS, ((length - 1) // 8) + 1, height // 32, width // 32


def _roll_modalities(samples, video_shape, shift, inverse=False):
    if shift == 0:
        return samples

    direction = shift if inverse else -shift
    if samples.ndim == 5:
        if tuple(samples.shape[1:]) != tuple(video_shape):
            raise ValueError(
                "LTX Mobius sampler received an unexpected video latent shape: "
                f"{tuple(samples.shape[1:])}, expected {tuple(video_shape)}"
            )
        return torch.roll(samples, shifts=direction, dims=2)

    if samples.ndim != 3 or samples.shape[1] != 1:
        raise ValueError(
            "LTX Mobius sampler expects a video latent or a packed LTX audio/video latent"
        )

    video_elements = math.prod(video_shape)
    if samples.shape[-1] < video_elements:
        raise ValueError(
            "Packed LTX latent is smaller than the configured video dimensions"
        )

    video = samples[..., :video_elements].reshape(
        samples.shape[0], *video_shape
    )
    video = torch.roll(video, shifts=direction, dims=2).reshape(
        samples.shape[0], 1, video_elements
    )

    packed_audio = samples[..., video_elements:]
    if packed_audio.numel() == 0:
        return video

    audio_frame_size = AUDIO_CHANNELS * AUDIO_FREQUENCY_BINS
    if packed_audio.shape[-1] % audio_frame_size:
        raise ValueError(
            "Packed LTX audio latent does not match the expected 8x16 layout"
        )

    audio_frames = packed_audio.shape[-1] // audio_frame_size
    audio_shift = round(shift * audio_frames / video_shape[1]) % audio_frames
    audio_direction = audio_shift if inverse else -audio_shift
    audio = packed_audio.reshape(
        samples.shape[0],
        AUDIO_CHANNELS,
        audio_frames,
        AUDIO_FREQUENCY_BINS,
    )
    audio = torch.roll(audio, shifts=audio_direction, dims=2).reshape(
        samples.shape[0], 1, packed_audio.shape[-1]
    )
    return torch.cat((video, audio), dim=-1)


class _MobiusModel:
    def __init__(
        self,
        model,
        video_shape,
        total_steps,
        shift_skip,
        start_percent,
        end_percent,
    ):
        self.model = model
        self.video_shape = video_shape
        self.total_steps = max(1, total_steps)
        self.shift_skip = shift_skip % video_shape[1]
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.call_index = 0
        self.shift = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, samples, sigma, **extra_args):
        denominator = max(1, self.total_steps - 1)
        progress = min(self.call_index, denominator) / denominator
        active = self.start_percent <= progress <= self.end_percent
        shift = self.shift if active else 0

        shifted = _roll_modalities(samples, self.video_shape, shift)
        denoised = self.model(shifted, sigma, **extra_args)
        denoised = _roll_modalities(
            denoised, self.video_shape, shift, inverse=True
        )

        if active:
            self.shift = (self.shift + self.shift_skip) % self.video_shape[1]
        self.call_index += 1
        return denoised


def sample_ltx_mobius(
    model,
    noise,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    width=768,
    height=512,
    length=97,
    shift_skip=4,
    start_percent=0.0,
    end_percent=1.0,
):
    video_shape = latent_video_shape(width, height, length)
    mobius_model = _MobiusModel(
        model=model,
        video_shape=video_shape,
        total_steps=len(sigmas) - 1,
        shift_skip=shift_skip,
        start_percent=start_percent,
        end_percent=end_percent,
    )
    return k_diffusion_sampling.sample_euler(
        mobius_model,
        noise,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
    )


class LTXMobiusSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 768, "min": 64, "max": 8192, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 64, "max": 8192, "step": 32},
                ),
                "length": (
                    "INT",
                    {"default": 97, "min": 9, "max": 4096, "step": 8},
                ),
                "shift_skip": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Latent frames shifted after each denoising step.",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = (
        "Training-free cyclic latent shifting based on Mobius. "
        "Video and LTX audio latents are shifted together."
    )

    def build(
        self,
        width,
        height,
        length,
        shift_skip,
        start_percent,
        end_percent,
    ):
        if start_percent > end_percent:
            raise ValueError("start_percent must not exceed end_percent")
        latent_video_shape(width, height, length)
        sampler = comfy.samplers.KSAMPLER(
            sample_ltx_mobius,
            extra_options={
                "width": width,
                "height": height,
                "length": length,
                "shift_skip": shift_skip,
                "start_percent": start_percent,
                "end_percent": end_percent,
            },
        )
        return (sampler,)


class LTXLoopDecodeTiled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tile_size": (
                    "INT",
                    {
                        "default": 768,
                        "min": 64,
                        "max": 4096,
                        "step": 32,
                    },
                ),
                "overlap": (
                    "INT",
                    {"default": 64, "min": 0, "max": 4096, "step": 32},
                ),
                "temporal_size": (
                    "INT",
                    {"default": 4096, "min": 8, "max": 4096, "step": 4},
                ),
                "temporal_overlap": (
                    "INT",
                    {"default": 4, "min": 4, "max": 4096, "step": 4},
                ),
                "context_latents": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Tail/head latent frames decoded around the loop boundary.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent/video"
    DESCRIPTION = (
        "Decodes a cyclic LTX latent with tail context before frame zero, "
        "then crops the central cycle."
    )

    def decode(
        self,
        samples,
        vae,
        tile_size,
        overlap,
        temporal_size,
        temporal_overlap,
        context_latents,
    ):
        latent = samples["samples"]
        if latent.ndim != 5:
            raise ValueError("LTX loop decode expects a five-dimensional video latent")

        context = min(context_latents, latent.shape[2] - 1)
        wrapped = torch.cat(
            (latent[:, :, -context:], latent, latent[:, :, :context]),
            dim=2,
        )

        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_size // 2

        temporal_compression = vae.temporal_compression_decode() or 1
        decode_temporal_size = max(2, temporal_size // temporal_compression)
        decode_temporal_overlap = max(
            1,
            min(
                decode_temporal_size // 2,
                temporal_overlap // temporal_compression,
            ),
        )
        spatial_compression = vae.spacial_compression_decode()
        images = vae.decode_tiled(
            wrapped,
            tile_x=tile_size // spatial_compression,
            tile_y=tile_size // spatial_compression,
            overlap=overlap // spatial_compression,
            tile_t=decode_temporal_size,
            overlap_t=decode_temporal_overlap,
        )

        start = context * temporal_compression
        frame_count = (latent.shape[2] - 1) * temporal_compression + 1
        images = images[:, start : start + frame_count]
        return (
            images.reshape(
                -1,
                images.shape[-3],
                images.shape[-2],
                images.shape[-1],
            ),
        )


class LTXLoopAudioSeam:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "blend_ms": (
                    "INT",
                    {
                        "default": 120,
                        "min": 0,
                        "max": 2000,
                        "step": 10,
                    },
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "blend"
    CATEGORY = "audio"
    DESCRIPTION = (
        "Matches the first and last audio sample with a smooth boundary correction "
        "without changing duration."
    )

    def blend(self, audio, blend_ms):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        sample_count = waveform.shape[-1]
        blend_samples = min(
            round(sample_rate * blend_ms / 1000),
            sample_count // 4,
        )
        if blend_samples < 2:
            return (audio,)

        output = waveform.clone()
        progress = torch.linspace(
            0.0,
            1.0,
            blend_samples,
            device=waveform.device,
            dtype=waveform.dtype,
        )
        smooth = progress * progress * (3.0 - 2.0 * progress)
        view_shape = [1] * waveform.ndim
        view_shape[-1] = blend_samples
        smooth = smooth.reshape(view_shape)

        midpoint = (waveform[..., :1] + waveform[..., -1:]) * 0.5
        output[..., :blend_samples] += (
            midpoint - waveform[..., :1]
        ) * (1.0 - smooth)
        output[..., -blend_samples:] += (
            midpoint - waveform[..., -1:]
        ) * smooth

        result = audio.copy()
        result["waveform"] = output
        return (result,)


NODE_CLASS_MAPPINGS = {
    "SetImageSize": LTXSetImageSize,
    "LTXSetImageSize": LTXSetImageSize,
    "LTXMobiusSampler": LTXMobiusSampler,
    "LTXLoopDecodeTiled": LTXLoopDecodeTiled,
    "LTXLoopAudioSeam": LTXLoopAudioSeam,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SetImageSize": "Set Image Size",
    "LTXSetImageSize": "Set Image Size",
    "LTXMobiusSampler": "LTX Mobius Sampler",
    "LTXLoopDecodeTiled": "LTX Loop Decode",
    "LTXLoopAudioSeam": "LTX Loop Audio Seam",
}
