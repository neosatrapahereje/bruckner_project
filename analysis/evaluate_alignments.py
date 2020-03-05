import numpy as np
import matplotlib.pyplot as plt

pairs = [('../align_norr/alignment_p_Furtwängler_-_1942_(Adagio)_r_Norrington_-_2008_(Adagio)_beat_times.txt',
          './furt_beats.txt'),
         ('../align_norr_hchroma/alignment_p_Furtwängler_-_1942_(Adagio)_r_Norrington_-_2008_(Adagio)_beat_times.txt',
          './furt_beats.txt'),
         ('../align_norr_pcchroma/alignment_p_Furtwängler_-_1942_(Adagio)_r_Norrington_-_2008_(Adagio)_beat_times.txt',
          './furt_beats.txt'),
         ('../align_norr_lin_spectrogram/alignment_p_Furtwängler_-_1942_(Adagio)_r_Norrington_-_2008_(Adagio)_beat_times.txt',
          './furt_beats.txt'),
         ('../align_norr_log_spectrogram/alignment_p_Furtwängler_-_1942_(Adagio)_r_Norrington_-_2008_(Adagio)_beat_times.txt',
          './furt_beats.txt'),
         ('../align_norr_mel_spectrogram/alignment_p_Furtwängler_-_1942_(Adagio)_r_Norrington_-_2008_(Adagio)_beat_times.txt',
          './furt_beats.txt'),
         ('../align_furt/alignment_p_Norrington_-_2008_(Adagio)_r_Furtwängler_-_1942_(Adagio)_beat_times.txt',
          './norr_beats.txt'),
         ('../align_furt_hchroma/alignment_p_Norrington_-_2008_(Adagio)_r_Furtwängler_-_1942_(Adagio)_beat_times.txt',
          './norr_beats.txt'),
         ('../align_furt_pcchroma/alignment_p_Norrington_-_2008_(Adagio)_r_Furtwängler_-_1942_(Adagio)_beat_times.txt',
          './norr_beats.txt'),
         ('../align_furt_lin_spectrogram/alignment_p_Norrington_-_2008_(Adagio)_r_Furtwängler_-_1942_(Adagio)_beat_times.txt',
          './norr_beats.txt'),
         ('../align_furt_log_spectrogram/alignment_p_Norrington_-_2008_(Adagio)_r_Furtwängler_-_1942_(Adagio)_beat_times.txt',
          './norr_beats.txt'),
         ('../align_furt_mel_spectrogram/alignment_p_Norrington_-_2008_(Adagio)_r_Furtwängler_-_1942_(Adagio)_beat_times.txt',
          './norr_beats.txt'),
]


results = []
beats = []
for ea_fn, ha_fn in pairs:

    ea = np.loadtxt(ea_fn)
    ha = np.loadtxt(ha_fn)

    diff = abs(ea - ha[:, 0])
    medaa = np.median(diff)
    maa = np.mean(diff)
    std = np.std(diff)
    results.append((medaa, maa, std))
    
    
