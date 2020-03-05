import numpy as np
import partitura
import re

# maximal IOI between notes in the chord
IOI_THRESHOLD = 0.03
# maximal IOI of the first to the last note in the chord
CHORD_THRESHOLD = 0.1

ptt = re.compile('alignment_p_(?P<perf_name>.*)_r_(?P<ref_name>.*).txt')


def get_slice_times(fn, out_file, ioi_threshold=IOI_THRESHOLD,
                    chord_threshold=CHORD_THRESHOLD,
                    check_notes=False):

    # Load Midi File
    mf = partitura.load_performance_midi(fn, merge_tracks=True)

    # Get note information
    notes = np.column_stack([mf.note_array['p_onset'],
                             mf.note_array['p_duration'],
                             mf.note_array['pitch'],
                             mf.note_array['velocity']])

    # sort notes by performed onset time
    notes = notes[np.argsort(notes[:, 0])]
    # First chord includes the first note
    chords = [[notes[0]]]

    # Initialize previous onset time
    prev_onset = notes[0, 0]

    for i in range(1, len(notes)):

        # Current onset time
        c_onset = notes[i, 0]

        # IOI between consecutive notes
        ioi = c_onset - prev_onset
        # IOI since the beginning of the chord
        ioi_cs = c_onset - chords[-1][0][0]

        # Conditions for belonging to the chord
        if ioi <= ioi_threshold and ioi_cs <= chord_threshold:

            chords[-1].append(notes[i])

        else:
            # Otherwise start a new chord
            chords.append([notes[i]])

        prev_onset = c_onset

    chords = [np.vstack(c) for c in chords]

    if check_notes:
        # check that all of the notes are included
        notes_in_chords = sum([len(c) for c in chords])

        assert notes_in_chords == len(notes)

    # Get slice time as the mean onset
    onset_times = np.array([np.mean(c[:, 0]) for c in chords])
    # Save slice times
    np.savetxt(out_file, onset_times)

    return onset_times

def find_nearest(array,value):
    """
    From https://stackoverflow.com/a/26026189
    """
    idx = np.clip(np.searchsorted(array, value, side="left"),
                  0, len(array) - 1)
    
    idx = idx - (np.abs(value - array[idx-1]) < np.abs(value - array[idx]))
    return idx

def align_onsets(score_alignment_fn, score_onsets_fn):
    """
    Create timing file for Sonic Visualizer from alignment
    """
    performer = ptt.search(score_alignment_fn).group('perf_name')

    try:
        score_onsets = np.loadtxt(score_onsets_fn, skiprows=0)
    except:
        score_onsets = np.loadtxt(score_onsets_fn, skiprows=1)

    score_alignment = np.loadtxt(score_alignment_fn, delimiter='\t')
    score_frames = score_alignment[:, 1]
    perf_frames = score_alignment[:, 0]

    frames_to_onset_idxs = np.unique(find_nearest(score_frames, score_onsets))

    performance_onset_times = np.unique(perf_frames[frames_to_onset_idxs])

    valid_idxs = np.where(np.diff(performance_onset_times) > 0.3)[0] + 1
    performance_onset_times = performance_onset_times[valid_idxs]

    np.savetxt('{0}_onsets.txt'.format(performer), performance_onset_times)

    return performance_onset_times


if __name__ == '__main__':

    import os
    import glob
    
    score_alignment_fn = '../pairwise_alignments_score/alignment_p_Celibidache_-_1990_(Adagio)_r_Bruckner_7_Adagio.mp3.txt'

    score_onsets_fn = '../score/onsets'

    filenames = glob.glob(os.path.join('../pairwise_alignments_score/', 'alignment*.txt'))

    for fn in filenames:
        align_onsets(fn, score_onsets_fn)
