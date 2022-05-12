import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import os
import glob
from scipy.signal import butter, lfilter, hamming

# Define filtering function to remove noise from signal
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = lfilter(b, a, data)
    return y

# Define two functions to cut or pad audio samples as necessary
# This ensures all audio samples are the same size
def cut_if_necessary(signal, desired_length):
    if len(signal) > desired_length:
        signal = signal[:desired_length]
    return signal

def right_pad_if_necessary(signal, desired_length):
    length_signal = len(signal)
    if length_signal < desired_length:
        num_missing_samples = desired_length - length_signal
        signal = np.pad(signal, num_missing_samples)
    return signal

# Define a function to retrieve the amplitude envelope of each frame
# Librosa does not have a function for this
def amplitude_envelope(signal, frame_size, hop_length):
    return np.array([max(signal[i:i+frame_size]) for i in range(0, signal.size, hop_length)])

# Below function averages every three sample points into a singular point
# Used to align the audio data frequency with facial data frequency
def average_samples(csv, samples=3):
    # csv is path to csv file to be altered
    # samples is the number of samples to be averaged.
    # i.e. if samples = 3, every 3 rows will be averaged into a single row
    # the returned table will then be 3 times smaller than the original
    data = np.genfromtxt(csv, dtype=float, delimiter=',')
    # setting up a blank array that the averages can later be appended to
    width = data.shape[1]
    new_data = np.zeros((1,width))

    for i in range(0, len(data)-samples, samples):
        total = np.zeros((1,width))   # reset total after each iteration
        if i % 10000 == 0:
            percent_done = (i/len(data)) * 100
            print(f"{percent_done:.2f}% complete")

        for j in range(0, samples):
            total += data[i+j,:]
        average = total / samples
        new_data = np.append(new_data, average, axis=0)
    # first row is entirely zeros, so return second row onwards
    print(data.shape)
    print(new_data.shape)
    return new_data[1:,:]


# Main function that extracts all audio features
# Add/remove functions from here to amend training features
# for audio only: used frame size = 1024 and hop length = 512
def extract_audio_features(file_path, frame_size=1024, hop_length=512, n_mfccs=5):
    # Load Audio Signal and apply bandpass filter
    signal, sr = librosa.load(file_path, sr=None)
    signal = butter_bandpass_filter(signal, 80, 18000, sr, order=5)  # bandpass 80Hz-18kHz
    signal = librosa.effects.preemphasis(signal, coef=0.97)


    # Ensure audio is the correct length of samples
    desired_num_samples = 154113
    signal = cut_if_necessary(signal, desired_num_samples)
    signal = right_pad_if_necessary(signal, desired_num_samples)

    # extract features
    stft = librosa.stft(signal, n_fft=frame_size,
                                    hop_length=hop_length,
                                    window='hamming')
    rms = librosa.feature.rms(y=None,
                                    S=stft,
                                    frame_length=frame_size,
                                    hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=signal,
                                    frame_length=frame_size,
                                    hop_length=hop_length,
                                    center=True)
    mfccs = librosa.feature.mfcc(y=signal,
                                    sr=sr,
                                    S=None,
                                    n_mfcc=n_mfccs,
                                    hop_length=hop_length)
    sp_rolloff = librosa.feature.spectral_rolloff(y=signal,
                                    sr=sr,
                                    hop_length=hop_length)
    sp_centroid = librosa.feature.spectral_centroid(y=signal,
                                    sr=sr,
                                    hop_length=hop_length)
    sp_contrast = librosa.feature.spectral_contrast(y=signal,
                                    sr=sr,
                                    hop_length=hop_length)
    sp_bandwidth = librosa.feature.spectral_bandwidth(y=signal,
                                    sr=sr,
                                    hop_length=hop_length)

    # reshape features before appending
    #ae = ae.reshape([-1, 1])
    rms = rms.reshape([-1, 1])
    zcr = zcr.reshape([-1, 1])
    mfccs = mfccs.reshape([-1, n_mfccs])
    sp_rolloff = sp_rolloff.reshape([-1, 1])
    sp_centroid = sp_centroid.reshape([-1, 1])
    sp_contrast = sp_contrast.reshape([-1, 7])
    sp_bandwidth = sp_bandwidth.reshape([-1, 1])
    #sp_flatness = sp_flatness.reshape([-1, 1])

    # add all features to "Features" array
    #features = np.array(ae)
    features = np.array(rms)
    features = np.append(features, zcr, axis=1)
    features = np.append(features, mfccs, axis=1)
    features = np.append(features, sp_rolloff, axis=1)
    features = np.append(features, sp_centroid, axis=1)
    features = np.append(features, sp_contrast, axis=1)
    features = np.append(features, sp_bandwidth, axis=1)
    #features = np.append(features, sp_flatness, axis=1)

    return features[:300,:]


def get_all_audio_features(directory, frame_size, hop_length, n_mfccs):
    # takes all audio files in a given directory
    # and applies "extract audio features" on them
    os.chdir(directory)
    extension = 'wav'

    # Gather all filenames
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    # apply function
    data = np.empty([1,n_mfccs+12], dtype=int)
    for f in all_filenames:
        print(f)
        #audio_dict["file%s" % f] = extract_audio_features(f, frame_size, hop_length, n_mfccs)
        data = np.append(data, extract_audio_features(f, frame_size, hop_length, n_mfccs), axis=0)
    return data

n_mfccs = 12
frame_size = 1024
hop_length = 512
directory = r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\"
test_or_train = r"test\\"
emotion_list = ['neutral_1', 'calm_2', 'happy_3', 'sad_4', 'angry_5', 'fearful_6', 'disgust_7', 'surprised_8']

# Now functions are called to create 8 separate dictionaries, one per emotion
'''for i, emotion in enumerate(emotion_list):
    data = get_all_audio_features(directory=f"{directory}{test_or_train}{emotion}",
                                  frame_size=frame_size,
                                  hop_length=hop_length,
                                  n_mfccs=n_mfccs)
    data = np.insert(data, 0, i, axis=1)
    np.savetxt(f"{directory}{emotion}.csv")'''

neutral_1 = get_all_audio_features(directory=r"audio\\test\\1_neutral",
                       frame_size=frame_size,
                       hop_length=hop_length,
                       n_mfccs=n_mfccs)
neutral_1 = np.insert(neutral_1, 0, 1, axis=1)
np.savetxt(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\1_neutral_data.csv", neutral_1, delimiter=",")

calm_2 = get_all_audio_features(directory=r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\test\\2_calm",
                       frame_size=frame_size,
                       hop_length=hop_length,
                       n_mfccs=n_mfccs)
calm_2 = np.insert(calm_2, 0, 2, axis=1)
np.savetxt(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\2_calm_data.csv", calm_2, delimiter=",")

happy_3 = get_all_audio_features(directory=r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\test\\3_happy",
                       frame_size=frame_size,
                       hop_length=hop_length,
                       n_mfccs=n_mfccs)
happy_3 = np.insert(happy_3, 0, 3, axis=1)
np.savetxt(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\3_happy_data.csv", happy_3, delimiter=",")

sad_4 = get_all_audio_features(directory=r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\test\\4_sad",
                       frame_size=frame_size,
                       hop_length=hop_length,
                       n_mfccs=n_mfccs)
sad_4 = np.insert(sad_4, 0, 4, axis=1)
np.savetxt(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\4_sad_data.csv", sad_4, delimiter=",")

angry_5 = get_all_audio_features(directory=r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\test\\5_angry",
                       frame_size=frame_size,
                       hop_length=hop_length,
                       n_mfccs=n_mfccs)
angry_5 = np.insert(angry_5, 0, 5, axis=1)
np.savetxt(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\5_angry_data.csv", angry_5, delimiter=",")

fearful_6 = get_all_audio_features(directory=r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\test\\6_fearful",
                       frame_size=frame_size,
                       hop_length=hop_length,
                       n_mfccs=n_mfccs)
fearful_6 = np.insert(fearful_6, 0, 6, axis=1)
np.savetxt(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\6_fearful_data.csv", fearful_6, delimiter=",")

disgust_7 = get_all_audio_features(directory=r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\test\\7_disgust",
                       frame_size=frame_size,
                       hop_length=hop_length,
                       n_mfccs=n_mfccs)
disgust_7 = np.insert(disgust_7, 0, 7, axis=1)
np.savetxt(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\7_disgust_data.csv", disgust_7, delimiter=",")

surprised_8 = get_all_audio_features(directory=r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\test\\8_surprised",
                       frame_size=frame_size,
                       hop_length=hop_length,
                       n_mfccs=n_mfccs)
surprised_8 = np.insert(surprised_8, 0, 8, axis=1)
np.savetxt(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\8_surprised_data.csv", surprised_8, delimiter=",")

all_data = neutral_1
all_data = np.append(all_data, calm_2, axis=0)
all_data = np.append(all_data, happy_3, axis=0)
all_data = np.append(all_data, sad_4, axis=0)
all_data = np.append(all_data, angry_5, axis=0)
all_data = np.append(all_data, fearful_6, axis=0)
all_data = np.append(all_data, disgust_7, axis=0)
all_data = np.append(all_data, surprised_8, axis=0)
np.savetxt(r"C:\\Users\\darre\\PycharmProjects\\Emotion_Classifier\\audio\\all_data.csv", all_data, delimiter=",")

print("Done")