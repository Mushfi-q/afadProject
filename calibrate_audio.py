import torch
import soundfile as sf
import os

def get_variance(audio_path):
    data, sample_rate = sf.read(audio_path)
    waveform = torch.tensor(data).float()
    if len(waveform.shape) > 1:
        waveform = torch.mean(waveform, dim=1, keepdim=True).t()
    else:
        waveform = waveform.unsqueeze(0)
    window_size = 1024
    hop_length = 512
    pad_size = window_size // 2
    padded_waveform = torch.nn.functional.pad(waveform, (pad_size, pad_size))
    windows = padded_waveform.unfold(-1, window_size, hop_length)
    rms_energy = torch.sqrt(torch.mean(windows**2, dim=-1))
    if torch.max(rms_energy) > 0:
        rms_energy = rms_energy / torch.max(rms_energy)
    return torch.var(rms_energy).item()

print("Analyzing Real Dataset:")
real_dir = "dataset/voice_dataset/real"
for f in os.listdir(real_dir):
    if f.endswith(".wav"):
        v = get_variance(os.path.join(real_dir, f))
        print(f"{f}: {v:.6f}")

print("\nAnalyzing Fake Dataset:")
fake_dir = "dataset/voice_dataset/deepfake"
for f in os.listdir(fake_dir):
    if f.endswith(".wav"):
        v = get_variance(os.path.join(fake_dir, f))
        print(f"{f}: {v:.6f}")
