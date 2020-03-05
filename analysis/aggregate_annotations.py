import numpy as np
import os
import glob
import partitura
from scipy.interpolate import interp1d

from utils import BeatMap

score = partitura.load_score_midi('../score/Bruckner_7_Adagio.mid', estimate_voice_info=False)
midi_score = partitura.load_performance_midi('../score/Bruckner_7_Adagio.mid', merge_tracks=True)

if not isinstance(score, (partitura.score.PartGroup, partitura.score.Part)):
    pg = partitura.score.PartGroup()
    pg.children = score
    score = pg


score_note_array = score.note_array
midi_note_array = midi_score.note_array

unique_s_onsets = np.unique(score_note_array['onset'])
unique_m_onsets = np.unique(midi_note_array['p_onset'])


midi_beat_map = BeatMap(input_times=unique_m_onsets,
                        output_times=unique_s_onsets,
                        i_min_time=unique_m_onsets.min(),
                        i_max_time=np.max(midi_note_array['p_onset'] + midi_note_array['p_duration']),
                        o_min_time=unique_s_onsets.min(),
                        o_max_time=np.max(score_note_array['onset'] + score_note_array['duration']))



data_dir = '../../bruckner_project_data/Sonic Visualiser timings/Furtwangler - 1942'

beat_files = glob.glob(os.path.join(data_dir, '*.csv'))

beats =[]
for fn in beat_files:

    b = np.loadtxt(fn, skiprows=1)
    beats.append(b)


mean_beats = np.median(beats, 0)


beat_idxs = np.where(np.mod(unique_s_onsets, 1) == 0)[0]
first_beat =  unique_s_onsets[beat_idxs[beat_idxs.argmin()]]
last_beat = unique_s_onsets[beat_idxs[beat_idxs.argmax()]]
score_beats = np.arange(0, 808 + 1, 0.5)


audio_beat_map = BeatMap(input_times=mean_beats,
                         output_times=score_beats[:len(mean_beats)],
                         i_min_time=0,
                         i_max_time=mean_beats.max() + 1,
                         o_min_time=0,
                         o_max_time=np.max(score_note_array['onset'] + score_note_array['duration']))


np.savetxt('furt_onsets.txt', audio_beat_map.inv_beat_map(unique_s_onsets))
np.savetxt('furt_beats.txt', np.column_stack((mean_beats, score_beats[:len(mean_beats)])))


