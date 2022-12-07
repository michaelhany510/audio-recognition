import pylab
from scipy.signal import find_peaks
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import read
import scipy

def makeOneChannel(audio):
    oneChannelAudio = []
    for point in audio:
        oneChannelAudio.append(int((point[0]+point[1])/2))
    return np.asarray( oneChannelAudio)

def getPeaksData(spectrogramArray):
    peaks, time_diff = find_peaks(spectrogramArray[0][0], height=0, width=0.5, distance=1.5)
    pylab.plot(spectrogramArray[0][0])
    pylab.plot(peaks, (spectrogramArray[0][0])[peaks], "x")
    pylab.plot(np.zeros_like(spectrogramArray[0][0]), "--", color="red")

def create_constellation(audio, Fs):
    if(len(audio.shape) > 1):
        audio = makeOneChannel(audio)
    
    # Parameters
    window_length_seconds = 0.5
    window_length_samples = int(window_length_seconds * Fs)
    window_length_samples += window_length_samples % 2
    num_peaks = 15
    
    # print(window_length_samples)

    # Pad the song to divide evenly into windows
    amount_to_pad = window_length_samples - audio.size % window_length_samples

    song_input = np.pad(audio, (0, amount_to_pad))

    # Perform a short time fourier transform
    # print(len(audio))
    frequencies, times, stft = scipy.signal.stft(song_input, Fs, nperseg=window_length_samples, nfft=window_length_samples, return_onesided=True)

    constellation_map = []

    for time_idx, window in enumerate(stft.T):
        # Spectrum is by default complex. 
        # We want real values only
        spectrum = abs(window)
        # Find peaks - these correspond to interesting features
        # Note the distance - want an even spread across the spectrum
        peaks, props = scipy.signal.find_peaks(spectrum, prominence=0, distance=200)

        # Only want the most prominent peaks
        # With a maximum of 15 per time slice
        n_peaks = min(num_peaks, len(peaks))
        # Get the n_peaks largest peaks from the prominences
        # This is an argpartition
        # Useful explanation: https://kanoki.org/2020/01/14/find-k-smallest-and-largest-values-and-its-indices-in-a-numpy-array/
        largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
        for peak in peaks[largest_peaks]:
            frequency = frequencies[peak]
            constellation_map.append([time_idx, frequency])

    return constellation_map


# constellation_map = create_constellation(song, Fs)

def create_hashes(constellation_map, song_id=None):
    hashes = {}
    # Use this for binning - 23_000 is slighlty higher than the maximum
    # frequency that can be stored in the .wav files, 22.05 kHz
    upper_frequency = 23_000 
    frequency_bits = 10

    # Iterate the constellation
    for idx, (time, freq) in enumerate(constellation_map):
        # Iterate the next 100 pairs to produce the combinatorial hashes
        # When we produced the constellation before, it was sorted by time already
        # So this finds the next n points in time (though they might occur at the same time)
        for other_time, other_freq in constellation_map[idx : idx + 100]: 
            diff = other_time - time
            # If the time difference between the pairs is too small or large
            # ignore this set of pairs
            if diff <= 1 or diff > 10:
                continue
            
            # Place the frequencies (in Hz) into a 1024 bins
            freq_binned = freq / upper_frequency * (2 ** frequency_bits)
            other_freq_binned = other_freq / upper_frequency * (2 ** frequency_bits)

            # Produce a 32 bit hash
            # Use bit shifting to move the bits to the correct location
            hash = int(freq_binned) | (int(other_freq_binned) << 10) | (int(diff) << 20)
            hashes[hash] = (time, song_id)
    
    # print(len(hashes))
    # print(hashes)
    return hashes

# # Quickly investigate some of the hashes produced
# hashes = create_hashes(constellation_map, 0)
# for i, (hash, (time, _)) in enumerate(hashes.items()):
#     if i > 10: 
#         break
#     print(f"Hash {hash} occurred at {time}")

import glob
from typing import List, Dict, Tuple
from tqdm import tqdm
import pickle


def updateDataBase():
    songs = glob.glob('gives error/*.wav')
    print(songs)
    song_name_index = {}
    database: Dict[int, List[Tuple[int, int]]] = {}

    # Go through each song, using where they are alphabetically as an id
    for index, filename in enumerate(sorted(songs)):
        print(index)
        song_name_index[index] = filename
        # Read the song, create a constellation and hashes
        Fs, audio_input = wavfile.read(filename)
        
        # print(audio_input.shape[0])
        # print(len(audio_input.shape))
        
        
        
        constellation = create_constellation(audio_input, Fs)
        hashes = create_hashes(constellation, index)

        # For each hash, append it to the list for this hash
        for hash, time_index_pair in hashes.items():
            if hash not in database:
                database[hash] = []
            database[hash].append(time_index_pair)

    print('end of DB')
    # Dump the database and list of songs as pickles
    with open("database.pickle", 'wb') as db:
        pickle.dump(database, db, pickle.HIGHEST_PROTOCOL)
    with open("song_index.pickle", 'wb') as songs:
        pickle.dump(song_name_index, songs, pickle.HIGHEST_PROTOCOL)

    # print(database)

updateDataBase()

# Load the database
database = pickle.load(open('database.pickle', 'rb'))
song_name_index = pickle.load(open("song_index.pickle", "rb"))
            
def score_hashes_against_database(hashes):
    matches_per_song = {}
    for hash, (sample_time, _) in hashes.items():
        if hash in database:
            matching_occurences = database[hash]
            for source_time, song_index in matching_occurences:
                if song_index not in matches_per_song:
                    matches_per_song[song_index] = []
                matches_per_song[song_index].append((hash, sample_time, source_time))
    
    scores = {}
    for song_index, matches in matches_per_song.items():
        song_scores_by_offset = {}
        for hash, sample_time, source_time in matches:
            delta = source_time - sample_time
            if delta not in song_scores_by_offset:
                song_scores_by_offset[delta] = 0
            song_scores_by_offset[delta] += 1
        max = (0, 0)
        for offset, score in song_scores_by_offset.items():
            if score > max[1]:
                max = (offset, score)
        
        scores[song_index] = max
    # Sort the scores for the user
    scores = list(sorted(scores.items(), key=lambda x: x[1][1], reverse=True)) 
    return scores

def score_hashes_against_database_NoTimeDelta(hashes):
    matches_per_song = {}
    for hash, (sample_time, _) in hashes.items():
        if hash in database:
            matching_occurences = database[hash]
            for source_time, song_index in matching_occurences:
                if song_index not in matches_per_song:
                    matches_per_song[song_index] = 0
                matches_per_song[song_index] += 1
    for song_id, num_matches in list(sorted(matches_per_song.items(), key=lambda x: x[1], reverse=True))[:10]:
        print(f"Song: {song_name_index[song_id]} - Matches: {num_matches}")


def print_top_five(file_name):
    # Load a short recording with some background noise
    Fs, audio_input = wavfile.read(file_name)
    # Create the constellation and hashes
    constellation = create_constellation(audio_input, Fs)
    # print(constellation)
    # print(len(constellation))
    hashes = create_hashes(constellation, None)
    # print(hashes)
    # print(len(hashes))
    scores = score_hashes_against_database(hashes)[:5]
    for song_id, score in scores:
        print(f"{song_name_index[song_id]}: Score of {score[1]} at {score[0]}")

# def print_top_five(file_name):
#     # Load a short recording with some background noise
#     Fs, audio_input = read(file_name)
#     # Create the constellation and hashes
#     constellation = create_constellation(audio_input, Fs)
#     hashes = create_hashes(constellation, None)

#     # score_hashes_against_database_NoTimeDelta(hashes)
#     scores = score_hashes_against_database_NoTimeDelta(hashes)[:5]
#     for song_id, score in scores:
#         print(f"{song_name_index[song_id]}: Score of {score[1]} at {score[0]}")

print("Recording 1:")
print_top_five("Michael open the door test.wav")

# print("\n\nRecording 2:")
# print_top_five("recording2.wav")