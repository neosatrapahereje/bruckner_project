import numpy as np
import madmom
import librosa
from scipy.fftpack import dct

def mfcc(fn_or_signal, num_bands=120, skip=20):
    # Calculate mfccs
    if isinstance(fn_or_signal, str):
        sig = madmom.audio.signal.Signal(file, sample_rate=44100, num_channels=1)
    else:
        sig = fn_or_signal
    sr = sig.sample_rate
    frame_size = 0.02 * sr
    hop_size = 0.01 * sr
    fs_mfcc = madmom.audio.signal.FramedSignal(sig, frame_size=frame_size,
                                               hop_size=hop_size)

    zeroPad = 2**0
    fft_size_mfcc = int(pow(2, np.round(np.log(frame_size * zeroPad)/np.log(2))))
    window_mfcc = np.hamming(fs_mfcc.frame_size + 1)[:-1]
    spec_mfcc = madmom.audio.spectrogram.Spectrogram(fs_mfcc, fft_size=fft_size_mfcc, window=window_mfcc)
    for i in range(spec_mfcc.shape[0]):
        spec_mfcc[i, :] -= np.min(spec_mfcc[i, :])
        if np.max(spec_mfcc[i, :]) !=  0:
            spec_mfcc[i, :] /= np.max(spec_mfcc[i, :])
            
    matMFCC = librosa.filters.mel(sr=sr, n_fft=fft_size_mfcc-1, n_mels=num_bands,
                                  fmin=0, fmax=sr/2,
                                  norm='slaney')
    mel_spec = np.dot(spec_mfcc, matMFCC.T)

    mfcc = dct(mel_spec, type=2, axis=1, norm='ortho')[:, skip:]

    for i in range(mfcc.shape[0]):
        if np.linalg.norm(mfcc[i, :]) == 0:
            mfcc[i, :] = np.ones(mfcc.shape[1]) * 1e-10

    mfcc_norm = mfcc.T  / np.linalg.norm(mfcc, axis=1)
    # mfcc_norm = mfcc_norm.T

    return mfcc_norm

