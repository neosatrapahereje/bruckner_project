import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

exp_pattrn = re.compile('(?P<ref>.+)_alignments_m_(?P<metric>.+)_f_(?P<features>.+)')

results_dir = '../exp_features_metrics/'

exp_dirs = glob.glob(os.path.join(results_dir,  '*_alignments'))

furt_beats = np.loadtxt('../bruckner_annotations/furt_beats.txt')
norr_beats = np.loadtxt('../bruckner_annotations/norr_beats.txt')

results_dict = dict()

for ref_dir in exp_dirs:

    m_f_dirs = glob.glob(os.path.join(ref_dir, '*_alignments_m*'))

    for ed in m_f_dirs:

        ref, metric, features = exp_pattrn.search(os.path.basename(ed)).groups()

        if ref == 'furt':
            ref_beats = furt_beats[:, 0]
            perf_beats_fn = os.path.join(ed, 'Norrington')
        elif ref == 'norr':
            ref_beats = norr_beats[:, 0]

        if ref not in results_dict:
            results_dict[ref] = dict(features=dict(metric=1))

        else:
            results_dict[ref][features][metric] = np.loadtxt()

        
        break
# pairs = [('../align_norr/alignment_p_Furtwängler_-_1942_(Adagio)_r_Norrington_-_2008_(Adagio)_beat_times.txt',
#           './furt_beats.txt'),
#          ('../align_norr_hchroma/alignment_p_Furtwängler_-_1942_(Adagio)_r_Norrington_-_2008_(Adagio)_beat_times.txt',
#           './furt_beats.txt'),
#          ('../align_norr_pcchroma/alignment_p_Furtwängler_-_1942_(Adagio)_r_Norrington_-_2008_(Adagio)_beat_times.txt',
#           './furt_beats.txt'),
#          ('../align_norr_lin_spectrogram/alignment_p_Furtwängler_-_1942_(Adagio)_r_Norrington_-_2008_(Adagio)_beat_times.txt',
#           './furt_beats.txt'),
#          ('../align_norr_log_spectrogram/alignment_p_Furtwängler_-_1942_(Adagio)_r_Norrington_-_2008_(Adagio)_beat_times.txt',
#           './furt_beats.txt'),
#          ('../align_norr_mel_spectrogram/alignment_p_Furtwängler_-_1942_(Adagio)_r_Norrington_-_2008_(Adagio)_beat_times.txt',
#           './furt_beats.txt'),
#          ('../align_furt/alignment_p_Norrington_-_2008_(Adagio)_r_Furtwängler_-_1942_(Adagio)_beat_times.txt',
#           './norr_beats.txt'),
#          ('../align_furt_hchroma/alignment_p_Norrington_-_2008_(Adagio)_r_Furtwängler_-_1942_(Adagio)_beat_times.txt',
#           './norr_beats.txt'),
#          ('../align_furt_pcchroma/alignment_p_Norrington_-_2008_(Adagio)_r_Furtwängler_-_1942_(Adagio)_beat_times.txt',
#           './norr_beats.txt'),
#          ('../align_furt_lin_spectrogram/alignment_p_Norrington_-_2008_(Adagio)_r_Furtwängler_-_1942_(Adagio)_beat_times.txt',
#           './norr_beats.txt'),
#          ('../align_furt_log_spectrogram/alignment_p_Norrington_-_2008_(Adagio)_r_Furtwängler_-_1942_(Adagio)_beat_times.txt',
#           './norr_beats.txt'),
#          ('../align_furt_mel_spectrogram/alignment_p_Norrington_-_2008_(Adagio)_r_Furtwängler_-_1942_(Adagio)_beat_times.txt',
#           './norr_beats.txt'),
# ]


# results = []
# beats = []
# for ea_fn, ha_fn in pairs:

#     ea = np.loadtxt(ea_fn)
#     ha = np.loadtxt(ha_fn)

#     diff = abs(ea - ha[:, 0])
#     medaa = np.median(diff)
#     maa = np.mean(diff)
#     std = np.std(diff)
#     results.append((medaa, maa, std))
    
    
