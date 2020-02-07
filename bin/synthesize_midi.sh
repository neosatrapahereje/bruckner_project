#!/bin/bash

# Synthesize a MIDI file using Fluidsynth

# Path to the script
scd="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
red="$(dirname $scd)"

# Midi file
midi_fn=$1
# Path to the audio file
out_file=$2

# Soundfont
soundfont="${red}/score/FluidR3_GM.sf2"

piece_name=$(basename "${midi_fn}" .mid)
wav_fn="${out_file}".wav
echo "Synthesizing ${piece_name}..."
# synthesize Wav
fluidsynth -n "${soundfont}" \
	   --fast-render="${wav_fn}" \
	   --audio-channels=1 \
	   --sample-rate=44100 \
	   --audio-file-format=s16 \
	   "${midi_fn}"

echo "Saving audio file to ${out_file}"
# Convert to mp3
sox "${wav_fn}" "${out_file}"
# Remove wav file
rm "${wav_fn}"
