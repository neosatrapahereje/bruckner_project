import numpy as np
from scipy.interpolate import interp1d
from madmom.audio.signal import FramedSignal

def find_nearest(array,value):
    """
    From https://stackoverflow.com/a/26026189
    """
    idx = np.clip(np.searchsorted(array, value, side="left"),
                  0, len(array) - 1)
    
    idx = idx - (np.abs(value - array[idx-1]) < np.abs(value - array[idx]))
    return idx

def frame_times_from_features(features):
    num_frames = features.spectrogram.stft.frames.num_frames
    fps = features.spectrogram.stft.frames.fps
    frame_times = np.arange(num_frames) / fps

    return frame_times

def score_beat_map(features, beats, score_beats=None):
    # Get frame times from FramedSignal
    if isinstance(features, FramedSignal):
        frame_times = frame_times_from_features(features)
    elif isinstance(features, np.ndarray):
        frame_times = features
    # Get indices of the beats in the frame_times
    beat_idxs = find_nearest(frame_times, beats)

    # Get beat times in frame_times
    beat_times = frame_times[beat_idxs]

    if score_beats is None:
        score_beats = np.arange(len(beats))

    beat_map = interp1d(beat_times, score_beats,
                        kind='linear',
                        bounds_error=False,
                        fill_value=(-1,
                                    score_beats.max()))
    index_map = interp1d(beat_idxs, score_beats,
                        kind='linear',
                        bounds_error=False,
                        fill_value=(-1,
                                    score_beats.max()))

    return beat_map, index_map


def get_beat_times(alignment, s_beats=None):

    if s_beats is None:
        beats = alignment[:, 2].astype(np.int)
    else:
        beats = alignment[:, 2]
    u_beats = np.unique(beats)
    u_beats.sort()

    if s_beats is None:
        u_beats = u_beats[u_beats >= 0]
    else:
        u_beats = u_beats[find_nearest(u_beats, s_beats)]

    beat_idxs = np.array([int(np.where(beats == u)[0].min())
                          for u in u_beats])

    beat_times = alignment[beat_idxs, 0]

    return beat_times


class BeatMap(object):
    def __init__(self, input_times, output_times, i_min_time, i_max_time, o_min_time, o_max_time):

        self.interp_fun = interp1d(input_times, output_times,
                                   kind='linear',
                                   bounds_error=False,
                                   fill_value='extrapolate')

        self.inv_fun = interp1d(output_times, input_times,
                                kind='linear',
                                bounds_error=False,
                                fill_value='extrapolate')

        self.i_min_time, self.i_max_time = i_min_time, i_max_time
        self.o_min_time, self.o_max_time = o_min_time, o_max_time

    def beat_map(self, input):
        return np.clip(self.interp_fun(input), self.i_min_time, self.i_max_time)

    def inv_beat_map(self, input):
        return np.clip(self.inv_fun(input), self.o_min_time, self.o_max_time)
