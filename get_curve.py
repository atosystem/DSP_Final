import sys

log_file = open(sys.argv[1],"r")

loss_tot=[]
err_tot=[]

ep = 0
ep_max = 25

for x in log_file.readlines() :
    if ("Epoch#" in x):
        if (ep>ep_max):
            break
        ep+=1
        l,e = x.split()[1].split(',')
        l = float(l.split('=')[1])
        e = float(e.split('=')[1])
        loss_tot.append(l)
        err_tot.append(e)
        # print(l,e)

import matplotlib.pyplot as plt
plt.figure()
# plt.plot( freq, np.abs(Y) )
plt.plot(loss_tot, label = "loss")
plt.plot(err_tot, label = "error")
plt.savefig('analyze/curve/{}.jpg'.format(sys.argv[1].split('/')[-1].replace('.txt','')))
plt.close()