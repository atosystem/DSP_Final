import librosa
import sys
import glob
import soundfile as sf
import os
import numpy as np
def readWave(filename, second_to_omit):

    wav, fs = librosa.load(filename)
    print(fs)
    # print(wav.shape)
    segs = librosa.effects.split(wav,40)
    # y,_ = librosa.effects.trim(wav, top_db=15) # denoise
    output_signals = []
    for (seq_in,seq_out) in segs:
        if (seq_out - seq_in) < fs*second_to_omit:
            # print("Omit")
            continue
        else:
            for i in range(seq_in, seq_out, fs*second_to_omit * 2):
                output_signals.append(wav[seq_in + i:seq_in + i + fs*second_to_omit * 2])
    # wav = np.append(y[0], y[1:] - 0.97 * y[:-1]) # # Preemphasis
    return output_signals

print("process file path : {}".format(sys.argv[1]))

allfiles = sorted(glob.glob(sys.argv[1]))
output_file = sys.argv[2]
print(output_file)
second_to_omit = 2
for i,f in enumerate(allfiles):
    print("processing : {}".format(f))
    for z,x in enumerate(readWave(f,second_to_omit)):
        x = x.T
        if(len(x) > 0):
            x = librosa.resample(x, 22050, 44100)
            # Signal normalization
            x=x/(np.max(np.abs(x))+1e-3)
            dir_path = "{}".format(f.split('/')[-1].replace('.wav',''))
            print(dir_path)
            if (not os.path.exists(os.path.join(output_file, dir_path))):
                print(os.path.join(output_file, dir_path))
                os.mkdir(os.path.join(output_file, dir_path))
            sf.write(os.path.join(output_file, dir_path,"{}.wav".format(z)),x,44100)
    # print(len(readWave(f)))


# print(readWave(sys.argv[1]))