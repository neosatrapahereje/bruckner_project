import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from statsmodels.nonparametric.smoothers_lowess import lowess



if __name__ == '__main__':

    ref_alignment = '../pairwise_alignments/alignment_p_Bruckner_7_Adagio.mp3_r_Karajan_-_1989__(Adagio).txt'
    score_alignment_fn = '../pairwise_alignments/alignment_p_Bruckner_7_Adagio.mp3_r_Karajan_-_1989__(Adagio).txt'

    score_alignment = np.loadtxt(score_alignment_fn, delimiter='\t')
    ref_alignment = np.loadtxt(ref_alignment, delimiter='\t')

    score_times = np.unique(score_alignment[:, 1])
    unique_score_idxs = [np.where(score_alignment[:, 1] == u)[0] for u in score_times]
    perf_times = np.array([np.mean(score_alignment[uix, 0]) for uix in unique_score_idxs])
    u_perf_times = np.unique(perf_times)

    # ref_times = np.unique(ref_alignment[:, 1])
    # unique_ref_idx = [np.where(ref_alignment[:, 1] == u)[0] for u in rer_times]
    # p_ref_times = np.array([np.mean(

    u_perf_time_idxs = [np.where(perf_times == u)[0] for u in u_perf_times]
    u_score_times = np.array([score_times[uix].mean() for uix in u_perf_time_idxs])

    score_ref_tempo_fun = interp1d(u_perf_times, u_score_times,
                                   kind='linear',
                                   bounds_error=False,
                                   fill_value='extrapolate')

    tempo_curve = np.diff(u_perf_times) / np.diff(u_score_times)

    filtered = lowess(tempo_curve, u_score_times[:-1], is_sorted=True, frac=0.020, it=3)

    plt.plot(filtered[:, 0], 60 / filtered[:, 1])
    plt.xlabel('Time (s)')
    plt.ylabel('Beats per minute')
    plt.savefig('tempo_Karajan.pdf')
    plt.clf()
    plt.close()
