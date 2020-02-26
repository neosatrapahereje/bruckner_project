import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from statsmodels.nonparametric.smoothers_lowess import lowess

def _get_unique(y):
    uy = np.unique(y)
    uy.sort()

    uidxs = [np.where(y == u) for u in uy]

    return uy, uidxs

def get_unique_timepoints(x, y, agg=np.mean):
    _uy, _uyids = _get_unique(y)

    _ux = np.array([agg(x[u]) for u in _uyids])

    ux, uxids = _get_unique(_ux)

    uy = np.array([agg(_uy[u]) for u in uxids])

    return ux, uy


    

    
    
    


if __name__ == '__main__':

    ref_alignment = '../pairwise_alignments/alignment_p_Bruckner_7_Adagio.mp3_r_Karajan_-_1989__(Adagio).txt'
    score_alignment_fn = '../pairwise_alignments/alignment_p_Bruckner_7_Adagio.mp3_r_Karajan_-_1989__(Adagio).txt'

    score_alignment = np.loadtxt(score_alignment_fn, delimiter='\t')
    ref_alignment = np.loadtxt(ref_alignment, delimiter='\t')

    rs_st, rs_pt = get_unique_timepoints(ref_alignment[:, 0], ref_alignment[:, 1])

    pr_st, pr_pt = get_unique_timepoints(score_alignment[:, 0], score_alignment[:, 1])

    ref_score_fun = interp1d(rs_st, rs_pt,
                       kind='linear',
                       bounds_error=False,
                       fill_value=(score_alignment[:, 1].min(), score_alignment[:, 1].max()))
    perf_score_fun = interp1d(pr_st, ref_score_fun(pr_pt),
                              kind='linear',
                              bounds_error=False,
                              fill_value=(score_alignment[:, 1].min(), score_alignment[:, 1].max()))

    

    
    plt.plot(rs_st, rs_pt)
    plt.plot(perf_score_fun(pr_st), pr_st)
    plt.show()

    # score_times = np.unique(ref_alignment[:, 1])
    # unique_score_idxs = [np.where(ref_alignment[:, 1] == u)[0] for u in score_times]
    # perf_times = np.array([np.mean(ref_alignment[uix, 0]) for uix in unique_score_idxs])
    # u_perf_times = np.unique(perf_times)

    # ref_times = np.unique(score_alignment[:, 1])
    # unique_ref_idx = [np.where(score_alignment[:, 1] == u)[0] for u in ref_times]
    # p_ref_times = np.array([np.mean(score_alignment[uix, 0]) for uix in unique_ref_idx])

    # u_perf_time_idxs = [np.where(perf_times == u)[0] for u in u_perf_times]
    # u_score_times = np.array([score_times[uix].mean() for uix in u_perf_time_idxs])

    # score_ref_tempo_fun = interp1d(u_perf_times, u_score_times,
    #                                kind='linear',
    #                                bounds_error=False,
    #                                fill_value='extrapolate')

    # perf_ref_fun = interp1d(p_ref_times, score_ref_tempo_fun(ref_times),
    #                         kind='linear',
    #                         bounds_error=False,
    #                         fill_value=(ref_times.min(), ref_times.max()))
    
    # tempo_curve2 = np.diff(perf_ref_fun(ref_times)) / np.diff(ref_times)

    # tempo_curve = np.diff(u_perf_times) / np.diff(u_score_times)


    # filtered = lowess(tempo_curve, u_score_times[:-1], is_sorted=True, frac=0.020, it=3)
    # filtered2 = lowess(tempo_curve2, ref_times[:-1], is_sorted=True, frac=0.020, it=3)
    

    # plt.plot(filtered[:, 0], 60 / filtered[:, 1])
    # plt.plot(filtered2[:, 0], 60 / filtered2[:, 1])
    # plt.show()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Beats per minute')
    # plt.savefig('tempo_Karajan.pdf')
    # plt.clf()
    # plt.close()
