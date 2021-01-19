import sys
sys.path.append('pytorch_cnn_visualizations/src/')

import matplotlib.pyplot as plt
import scipy.fftpack
# speaker_id.py
# Mirco Ravanelli 
# Mila - University of Montreal 

# July 2018

# Description: 
# This code performs a speaker_id experiments with SincNet.
 
# How to run it:
# python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg

import os
#import scipy.io.wavfile
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


import numpy as np
from dnn_models import MLP,flip
from dnn_models import SincNet as CNN 
from data_io import ReadList,read_conf,str_to_bool
from pytorch_cnn_visualizations.src.vanilla_backprop import vanilla
import  pytorch_cnn_visualizations.src.misc_functions
import matplotlib.pyplot as plt


def create_batches_rnd(batch_size,data_folder,wav_lst,N_snt,wlen,lab_dict,fact_amp):
    
  # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
  sig_batch=np.zeros([batch_size,wlen])
  lab_batch=np.zeros(batch_size)
    
  snt_id_arr=np.random.randint(N_snt, size=batch_size)

  rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)

  for i in range(batch_size):
      
    # select a random sentence from the list 
    #[fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
    #signal=signal.astype(float)/32768

    [signal, fs] = sf.read(data_folder+wav_lst[snt_id_arr[i]])

    # accesing to a random chunk
    snt_len=signal.shape[0]
    snt_beg=np.random.randint(snt_len-wlen-1) #randint(0, snt_len-2*wlen-1)
    snt_end=snt_beg+wlen

    channels = len(signal.shape)
    if channels == 2:
      print('WARNING: stereo to mono: '+data_folder+wav_lst[snt_id_arr[i]])
      signal = signal[:,0]
    
    sig_batch[i,:]=signal[snt_beg:snt_end]*rand_amp_arr[i]
    lab_batch[i]=lab_dict[wav_lst[snt_id_arr[i]]]
    
  inp=Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
  lab=Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())
    
  return inp,lab  



# Reading cfg file
options=read_conf()

#[data]
tr_lst=options.tr_lst
te_lst=options.te_lst
pt_file=options.pt_file
class_dict_file=options.lab_dict
data_folder=options.data_folder+'/'
output_folder=options.output_folder

#[windowing]
fs=int(options.fs)
cw_len=int(options.cw_len)
cw_shift=int(options.cw_shift)

#[cnn]
cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act=list(map(str, options.cnn_act.split(',')))
cnn_drop=list(map(float, options.cnn_drop.split(',')))


#[dnn]
fc_lay=list(map(int, options.fc_lay.split(',')))
fc_drop=list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp=str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp=str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm=list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm=list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act=list(map(str, options.fc_act.split(',')))

#[class]
class_lay=list(map(int, options.class_lay.split(',')))
class_drop=list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp=str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp=str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm=list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm=list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act=list(map(str, options.class_act.split(',')))


#[optimization]
lr=float(options.lr)
batch_size=int(options.batch_size)
N_epochs=int(options.N_epochs)
N_batches=int(options.N_batches)
N_eval_epoch=int(options.N_eval_epoch)
seed=int(options.seed)


# training list
wav_lst_tr=ReadList(tr_lst)
snt_tr=len(wav_lst_tr)

# test list
wav_lst_te=ReadList(te_lst)
snt_te=len(wav_lst_te)


# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder) 
    
    
# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
cost = nn.NLLLoss()

  
# Converting context and shift in samples
wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)

# Batch_dev
Batch_dev=128


# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
          'fs': fs,
          'cnn_N_filt': cnn_N_filt,
          'cnn_len_filt': cnn_len_filt,
          'cnn_max_pool_len':cnn_max_pool_len,
          'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
          'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
          'cnn_use_laynorm':cnn_use_laynorm,
          'cnn_use_batchnorm':cnn_use_batchnorm,
          'cnn_act': cnn_act,
          'cnn_drop':cnn_drop,          
          }

CNN_net=CNN(CNN_arch)
# CNN_net.cuda()

# Loading label dictionary
lab_dict=np.load(class_dict_file,allow_pickle=True).item()



DNN1_arch = {'input_dim': CNN_net.out_dim,
          'fc_lay': fc_lay,
          'fc_drop': fc_drop, 
          'fc_use_batchnorm': fc_use_batchnorm,
          'fc_use_laynorm': fc_use_laynorm,
          'fc_use_laynorm_inp': fc_use_laynorm_inp,
          'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
          'fc_act': fc_act,
          }

DNN1_net=MLP(DNN1_arch)
# DNN1_net.cuda()


DNN2_arch = {'input_dim':fc_lay[-1] ,
          'fc_lay': class_lay,
          'fc_drop': class_drop, 
          'fc_use_batchnorm': class_use_batchnorm,
          'fc_use_laynorm': class_use_laynorm,
          'fc_use_laynorm_inp': class_use_laynorm_inp,
          'fc_use_batchnorm_inp':class_use_batchnorm_inp,
          'fc_act': class_act,
          }


DNN2_net=MLP(DNN2_arch)
# DNN2_net.cuda()


if pt_file!='none':
   checkpoint_load = torch.load(pt_file)
   CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
   DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
   DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])


from SincNet import SincNet_Model
my_sinc_net = SincNet_Model(CNN_net, DNN1_net, DNN2_net)

file_for_input = "/home/makerspace/mks_users_home/dspfinal/training_data/data/train/linshan/1_trim_5.wav"
target = np.array([0])

# file_for_input = "/home/makerspace/mks_users_home/dspfinal/training_data/data/train/yun/1_1.wav"
# target = np.array([1])


[orig_signal, fs] = sf.read(file_for_input)

# wlen = 1 * fs

# my_sinc_net.CNN.input_dim = wlen

# my_sinc_net.cuda()

# slice first wlen ms
# print(len(orig_signal))
# exit()
all_altered_signal = []
# print(len(orig_signal))
# exit()
for i in range(0,len(orig_signal),wlen):
  print("processing segment #{} to #{}".format(i,i+wlen))#,end='\r')

  orig_signal_seg = orig_signal[i:i+wlen]

  if (not len(orig_signal_seg) == wlen):
    break

  orig_signal_seg = Variable(torch.from_numpy(orig_signal_seg).float().contiguous())
  # target=Variable(torch.from_numpy(target).float().cuda().contiguous())
  target = 0
  orig_signal_seg = orig_signal_seg.unsqueeze(0)
  # target = target.unsqueeze(0)

  # print(orig_signal.shape)
  # print(target.shape)
  # exit()
  # print(wlen)
  # exit()
  grad_arr = vanilla(my_sinc_net,input_data=[orig_signal_seg,target])
  # print("done vanilla")
  # print(grad_arr.shape)
  # print(orig_signal.detach().numpy().shape)
  orig_signal_seg = orig_signal_seg.detach().squeeze().numpy()
  # exit()
  # multiply grad_arr to input
  # print(grad_arr)
  # grad_arr = np.clip(grad_arr,0.5,None)
  # for x in range(len(grad_arr)):
  #   if grad_arr[x] < 3:
  #     grad_arr[x] = 0
  # grad_arr = grad_arr[0]

  # print(orig_signal_seg)
  # altered_signal = np.multiply(orig_signal_seg,grad_arr)
  # print(orig_signal_seg)
  # print(np.max(grad_arr))
  # print(altered_signal)
  # exit()
  # print(np.max(altered_signal))
  # exit()
  all_altered_signal.extend(grad_arr)


# print(all_altered_signal)
# exit()
# print(altered_signal.shape)
# print(grad_arr)


# all_altered_signal = all_altered_signal
# write the signal into file
sf.write("analyze/grad/{}_original.wav".format(file_for_input.split('/')[-1].replace('.wav','')),orig_signal,44100)
sf.write("analyze/grad/{}_grad_pure.wav".format(file_for_input.split('/')[-1].replace('.wav','')),all_altered_signal,44100)

print("Done writing wav files")
exit()

Y    = np.fft.fft(altered_signal)
freq = np.fft.fftfreq(wlen, 1.0 / fs)

# fig, ax = plt.subplots()
# ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
# plt.show()


plt.figure()
plt.plot( freq, np.abs(Y) )
plt.savefig('analyze/grad/fft.jpg')
plt.close()

exit()
