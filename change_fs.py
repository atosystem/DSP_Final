import soundfile as sf
import sys
import glob
import librosa

allfiles = sorted(glob.glob(sys.argv[1]))



for i,f in enumerate(allfiles):
    print("processing {}".format(f))
    [signal, fs] = sf.read(f)
    # trim front 20 sec
    signal = signal[20*fs:]
    sf.write("{}_trim.wav".format(i+1),signal,44100)
    # sf.write("{}_trim.wav".format(f),signal,44100)


# print(allfiles)
