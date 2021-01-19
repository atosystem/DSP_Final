import soundfile as sf
import sys


wav_file = sys.argv[1]

[signal, fs] = sf.read(wav_file)

print(fs)

# for 
