import numpy as np
import matplotlib
matplotlib.use('agg')
from alignment_features import mfcc
from librosa.sequence import dtw
import matplotlib.pyplot as plt

import os
import glob

import madmom

import time

def _perf_name(fn):
    return os.path.basename(fn).replace(' ', '_').replace('.flac', '')

if  __name__ == '__main__':
    data_dir = r'../data/'

    audio_files = glob.glob(os.path.join(data_dir, '*', '*.flac'))

    fn = audio_files[0]

    reference_fn = audio_files[1]

    sample_rate = int(44100 / 4)
    
    start = time.time()
    print('Loading performance')
    performance = madmom.audio.signal.Signal(fn, sample_rate=sample_rate,
                                             num_channels=1)[:sample_rate * 60 * 2]

    print('Loading reference')
    reference = madmom.audio.signal.Signal(reference_fn, sample_rate=sample_rate,
                                           num_channels=1)[:sample_rate * 60 * 2]

    print('extracting audio features from performance')
    perf_mfcc, perf_framed = mfcc(performance)

    print('extracting audio features from reference')
    ref_mfcc, ref_framed = mfcc(reference)

    perf_times = np.arange(perf_framed.num_frames) / perf_framed.fps
    ref_times = np.arange(ref_framed.num_frames) / ref_framed.fps

    fig, axes = plt.subplots(2)
    axes[0].imshow(perf_mfcc, aspect='auto',
               interpolation='nearest',
               origin='lowest')
    axes[1].imshow(ref_mfcc, aspect='auto',
               interpolation='nearest',
               origin='lowest')
    plt.savefig('features.pdf')
    # plt.show()
    plt.clf()
    plt.close()

    print('Estimating algnment')
    D, wp = dtw(perf_mfcc, ref_mfcc)
    wp = wp[::-1]
    plt.plot(perf_times[wp[:, 0]], ref_times[wp[:, 1]])
    plt.xlabel('Performance time (s)')
    plt.ylabel('Reference time (s)')
    plt.savefig('alignment.pdf')
    plt.clf()
    plt.close()

    end = time.time()

    print('alignment duration {0}'.format(end -start))

    alignment = np.column_stack((perf_times[wp[:, 0]], ref_times[wp[:, 1]]))
    out_fn = 'alignment_p_{0}_r_{1}.txt'.format(_perf_name(fn), _perf_name(reference_fn))
    np.savetxt(out_fn, alignment, delimiter='\t')

