import numpy as np
from alignment_features import mfcc
from librosa.sequence import dtw

import os
import glob

import madmom

if  __name__ == '__main__':
    data_dir = r'/Users/aae/Downloads/Bruckner-Mirage project files/'

    audio_files = glob.glob(os.path.join(data_dir, '*', '*.flac'))

    fn = audio_files[0]

    reference_fn = audio_files[1]

    sample_rate = 44100

    print('Loading performance')
    performance = madmom.audio.signal.Signal(fn, sample_rate, num_channels=1)

    print('Loading reference')
    reference = madmom.audio.signal.Signal(reference_fn, sample_rate, num_channels=1)

    print('extracting audio features from performance')
    perf_mfcc = mfcc(performance)

    print('extracting audio features from reference')
    ref_mfcc = mfcc(reference)

    print('Estimating alginment')
    D, wp = dtw(perf_mfcc, ref_mfcc)
    
    



