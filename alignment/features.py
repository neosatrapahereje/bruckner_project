# -*- coding: utf-8 -*-
import logging

import madmom
import numpy as np
import librosa

from madmom.audio import cepstrogram
from madmom.audio.filters import MelFilterbank

from scipy.fftpack import dct

LOGGER = logging.getLogger(__name__)

# Sample rate in Hz
SAMPLE_RATE = 44100
FRAME_SIZE = 0.1
HOP_SIZE = 0.05
DEFAULT_FEATURE_KWARGS = dict(
    features='mfcc_madmom',
    num_bands=120,
    skip=20
)


def extract_features(fn, features='mfcc',
                     feature_kwargs=DEFAULT_FEATURE_KWARGS,
                     sample_rate=SAMPLE_RATE,
                     frame_size=FRAME_SIZE,
                     hop_size=HOP_SIZE):

    signal = load_audio(fn, sample_rate=sample_rate)

    framed_signal = frame_signal(signal, frame_size=frame_size,
                                 hop_size=hop_size)

    return compute_features(framed_signal, features=features, **feature_kwargs)
    

def load_audio(fn_or_signal, sample_rate=SAMPLE_RATE):
    """
    Load audio file

    Parameters
    ----------
    fn_or_signal : str or madmom.audio.Signal
       File to be loaded as a signal.

    Returns
    -------
    sig: madmom.audio.Signal
       audio signal
    """
    if isinstance(fn_or_signal, str):
        LOGGER.debug('Loading audio file')
        sig = madmom.audio.signal.Signal(fn_or_signal, sample_rate=SAMPLE_RATE, num_channels=1)
    elif isinstance(fn_or_signal, madmom.audio.signal.Signal):
        sig = fn_or_signal

    return sig

def frame_signal(signal, frame_size=0.02, hop_size=0.01):
    """
    Creates a FramedSignal from a Signal
    """
    if not isinstance(signal, madmom.audio.signal.Signal):
        raise ValueError('Expected an instance of `madmom.audio.signal.Signal`, '
                         'but got {0}'.format(type(signal)))

    sample_rate = signal.sample_rate
    if frame_size < 1:
        frame_size = int(np.round(frame_size * sample_rate))

    if hop_size < 1:
        hop_size = int(np.round(hop_size * sample_rate))

    framed_signal = madmom.audio.signal.FramedSignal(signal=signal,
                                                     frame_size=frame_size,
                                                     hop_size=hop_size)

    LOGGER.debug('Frames: {0}\nFPS: {1}\nFrame size: {2}\nHop size: {3}'.format(
        framed_signal.num_frames, framed_signal.fps,
        framed_signal.hop_size, framed_signal.frame_size))

    return framed_signal


def compute_features(framed_signal, features='mfcc', *args, **kwargs):
    """
    Extract features from a FramedSignal

    Parameters
    ----------
    framed_signal : madmom.audio.signal.FramedSignal
       An instance of a framed signal.
    features : str
       Type of features. Additional arguments or keyword arguments
       are passed to the method

    Returns
    -------
    features : np.ndarray
        An array of size (num_frames, num_features) with the features
    frame_times : np.array
        An array with the times in seconds corresponding to each frame
    """
    frame_times = np.arange(framed_signal.num_frames) / framed_signal.fps

    feature_func = globals()[features]
    return feature_func(framed_signal, *args, **kwargs), frame_times
    # if features == 'mfcc':
    #     return mfcc(framed_signal, *args, **kwargs), frame_times
    # elif features == 'clp_chroma':
    #     return clp_chroma(framed_signal, *args, **kwargs), frame_times


def mfcc(framed_signal, num_bands=120, skip=20):
    """
    Mel Frequency Cepstral Coefficients
    
    Parameters
    ----------
    framed_signal : madmom.audio.signal.FramedSignal
       An instance of a framed signal.
    num_bands : int
       Number of frequency bands
    skip : int
       Number of MFCC to skip

    Returns
    -------
    mfcc_norm : np.ndarray
       An array of size (num_frames, num_features) with the features.
    """
    frame_size = framed_signal.frame_size
    hop_size = framed_signal.hop_size
    sample_rate = framed_signal.signal.sample_rate

    zeroPad = 2**0
    fft_size_mfcc = int(pow(2, np.round(np.log(frame_size * zeroPad)/np.log(2))))
    window_mfcc = np.hamming(framed_signal.frame_size + 1)[:-1]
    spec_mfcc = madmom.audio.spectrogram.Spectrogram(framed_signal,
                                                     fft_size=fft_size_mfcc,
                                                     window=window_mfcc)
    for i in range(spec_mfcc.shape[0]):
        spec_mfcc[i, :] -= np.min(spec_mfcc[i, :])
        if np.max(spec_mfcc[i, :]) !=  0:
            spec_mfcc[i, :] /= np.max(spec_mfcc[i, :])
            
    matMFCC = librosa.filters.mel(sr=sample_rate,
                                  n_fft=fft_size_mfcc-1,
                                  n_mels=num_bands,
                                  fmin=0, fmax=sample_rate/2,
                                  norm='slaney')
    mel_spec = np.dot(spec_mfcc, matMFCC.T)

    mfcc = dct(mel_spec, type=2, axis=1, norm='ortho')[:, skip:]

    for i in range(mfcc.shape[0]):
        if np.linalg.norm(mfcc[i, :]) == 0:
            mfcc[i, :] = np.zeros(mfcc.shape[1]) + 1e-10

    mfcc_norm = mfcc.T  / np.linalg.norm(mfcc, axis=1)

    return mfcc.astype(np.float32)


def pc_chroma(framed_signal, *args, **kwargs):

    frame_size = framed_signal.frame_size
    hop_size = framed_signal.hop_size
    sample_rate = framed_signal.signal.sample_rate

    zeroPad = 2**0
    fft_size = int(pow(2, np.round(np.log(frame_size * zeroPad)/np.log(2))))
    window = np.hamming(framed_signal.frame_size + 1)[:-1]
    spectrogram = madmom.audio.spectrogram.Spectrogram(framed_signal,
                                                     fft_size=fft_size,
                                                     window=window)

    chroma = madmom.audio.chroma.PitchClassProfile(spectrogram)

    return chroma.astype(np.float32)


def harmonic_chroma(framed_signal, *args, **kwargs):

    frame_size = framed_signal.frame_size
    hop_size = framed_signal.hop_size
    sample_rate = framed_signal.signal.sample_rate

    zeroPad = 2**0
    fft_size = int(pow(2, np.round(np.log(frame_size * zeroPad)/np.log(2))))
    window = np.hamming(framed_signal.frame_size + 1)[:-1]
    spectrogram = madmom.audio.spectrogram.Spectrogram(framed_signal,
                                                     fft_size=fft_size,
                                                     window=window)

    chroma = madmom.audio.chroma.HarmonicPitchClassProfile(spectrogram)

    return chroma.astype(np.float32)



def mfcc_madmom(framed_signal, num_bands=120, skip=20,
                *args, **kwargs):

    frame_size = framed_signal.frame_size
    hop_size = framed_signal.hop_size
    sample_rate = framed_signal.signal.sample_rate

    zeroPad = 2**0
    fft_size = int(pow(2, np.round(np.log(frame_size * zeroPad)/np.log(2))))
    window = np.hamming(framed_signal.frame_size + 1)[:-1]

    spectrogram = madmom.audio.spectrogram.FilteredSpectrogram(framed_signal,
                                                               filterbank=MelFilterbank,
                                                     fft_size=fft_size,
                                                     window=window)
 

    mfcc = cepstrogram.MFCC(spectrogram, num_bands=num_bands)

    return mfcc[:, skip:].astype(np.float32)
