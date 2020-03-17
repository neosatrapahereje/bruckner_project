import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

exp_pattrn = re.compile('(?P<ref>.+)_alignments_m_(?P<metric>.+)_f_(?P<features>.+)')
al_ptt = re.compile('alignment_p_(?P<perf>.+)_r_(?P<ref>.+)_beat_times.txt')

results_dir = '../exp_features_metrics/'

exp_dirs = glob.glob(os.path.join(results_dir,  '*_alignments'))

furt_beats = np.loadtxt('../bruckner_annotations/furt_beats.txt')
norr_beats = np.loadtxt('../bruckner_annotations/norr_beats.txt')

results_dict = dict()

for ref_dir in exp_dirs:

    m_f_dirs = glob.glob(os.path.join(ref_dir, '*_alignments_m*'))

    for ed in m_f_dirs:

        ref, metric, features = exp_pattrn.search(os.path.basename(ed)).groups()

        print(ref, metric, features)

        if ref == 'furt':
            ref_beats = furt_beats[:, 0]
            perf_beats_fn = os.path.join(ed, 'alignment_p_Norrington_-_2008_(Adagio)_r_Furtwängler_-_1942_(Adagio)_beat_times.txt')

        elif ref == 'norr':
            ref_beats = norr_beats[:, 0]
            perf_beats_fn = os.path.join(ed, 'alignment_p_Furtwängler_-_1942_(Adagio)_r_Norrington_-_2008_(Adagio)_beat_times.txt')

        perf_beats = np.loadtxt(perf_beats_fn)

        res = np.column_stack((perf_beats, ref_beats))

        if ref not in results_dict:
            results_dict[ref] = {features:{metric:res}}
        elif features not in results_dict[ref]:
            results_dict[ref][features] = {metric:res}
        else:
            results_dict[ref][features][metric] = res    
    

        print(results_dict[ref].keys())


metrics = ['euclidean', 'cosine', 'l1']
features = ['log_spectrogram', 'pc_chroma', 'mfcc', 'harmonic_chroma', 'lin_spectrogram', 'mel_spectrogram']
# aggregate by metric

results_by_metric = []
for ref in results_dict:
    for me in metrics:
        me_results = []
        for fe in features:
            diff = abs(results_dict[ref][fe][me][:, 0] - results_dict[ref][fe][me][:, 1])
            me_results.append(np.median(diff))
        results_by_metric.append(me_results)


results_by_metric = np.array(results_by_metric)        
    
