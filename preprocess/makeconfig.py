import sys
import glob
import os
from shutil import copyfile
import numpy as np

# python makeconfig.py ../../training_data/data/all/  ../../training_data/data/ 0.9

all_data_path = sys.argv[1] #ex ../../training_data/data/all/
outputpath = sys.argv[2] #datafolder # ex ../../training_data/data/
# output_datalist_path = sys.argv[3]
train_data_ratio = float(sys.argv[3]) #ex 0.9

speakers = ['linshan','yun']

print("All data path from {}".format(all_data_path))

# create speakers folders in traing adn testing
if not os.path.exists(os.path.join(outputpath,"train")):
    os.makedirs(os.path.join(outputpath,"train"))
if not os.path.exists(os.path.join(outputpath,"test")):
    os.makedirs(os.path.join(outputpath,"test"))

for spk in speakers:
    if not os.path.exists(os.path.join(outputpath,"train",spk)):
        os.makedirs(os.path.join(outputpath,"train",spk))
    if not os.path.exists(os.path.join(outputpath,"test",spk)):
        os.makedirs(os.path.join(outputpath,"test",spk))

scp_train = []
scp_test = []
label_data= {}



for i_spk,spk in enumerate(speakers):
    speaker_files = sorted(glob.glob(os.path.join(all_data_path,spk,'*.wav')))
    print("[{}]Speaker {} has {} files".format(i_spk,spk,len(speaker_files)))
    for file_index, f in enumerate(speaker_files):
        
        if (file_index<train_data_ratio*len(speaker_files)):
            # put inside training folder
            copyfile(f, os.path.join(outputpath,"train",spk,f.split('/')[-1]))
            scp_train.append(os.path.join("train",spk,f.split('/')[-1]))
            label_data[os.path.join("train",spk,f.split('/')[-1])]=i_spk
        else:
            # put inside testing folder
            copyfile(f, os.path.join(outputpath,"test",spk,f.split('/')[-1]))
            scp_test.append(os.path.join("test",spk,f.split('/')[-1]))
            label_data[os.path.join("test",spk,f.split('/')[-1])]=i_spk


def merge_two_dicts(x, y):
    """Given two dictionaries, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

# print("Saving dsp_all.scp")
# with open('dsp_all.scp', 'w') as f:
#     for item in scp_train + scp_test:
#         f.write("%s\n" % item)

output_scp_path = "../dsp_data_lists"
print("Saving dsp_train.scp")
with open(os.path.join(output_scp_path,'dsp_train.scp'), 'w') as f:
    for item in scp_train:
        f.write("%s\n" % item)

print("Saving dsp_test.scp")
with open(os.path.join(output_scp_path,'dsp_test.scp'), 'w') as f:
    for item in scp_test:
        f.write("%s\n" % item)

print("Saving dsp_labels.npy")
np.save(os.path.join(output_scp_path,'dsp_labels.npy'),label_data)


