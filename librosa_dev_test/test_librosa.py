import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load the audio with librosa
data, sampling_rate = librosa.load('audata/cats_dogs/train/cat/cat_125.wav')
print("Data loaded successfully!")

# Create the waveform figure
plt.figure(figsize=(12, 4))
librosa.display.waveshow(data, sr=sampling_rate, color='blue')
plt.title('Waveform')

# Create the spectrum figure
n_fft = 2048
ft = np.abs(librosa.stft(data[:n_fft], hop_length = n_fft+1))
plt.figure(figsize=(12, 4))
plt.plot(ft)
plt.title('Spectrum')
plt.xlabel('Frequency Bin')
plt.ylabel('Amplitude')

# Create the spectrogram figure
D = librosa.stft(data, n_fft=n_fft, hop_length=512)
D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
plt.figure(figsize=(12, 4))
librosa.display.specshow(D_db, sr=sampling_rate, hop_length=512, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')

# Create the MFCC figure
try:
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=13)
except TypeError as e:
    print(f"Error while computing MFCCs: {e}")
    # If there's a TypeError, try an alternative method to call mfcc
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=13)

plt.figure(figsize=(12, 4))
librosa.display.specshow(mfccs, sr=sampling_rate, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')

# Show all figures
plt.show()
