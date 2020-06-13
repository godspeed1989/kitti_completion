import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

def int2int(a):
    s = str(int(a))
    l = len(s)
    b = 1
    if l < 2:
        b = 0
    elif int(s[1]) < 5:
        b = 0
    ret = (int(s[0])+b) * (10**(l-1))
    return ret

d0=[\
[1, 23112, 29870],
[2, 20493, 28447],
[4, 8557, 14913],
[8, 3283, 6622],
[16, 1081, 2790],
[32, 275, 841],
[64, 201, 709]
]

d1=[\
[0.1, 913, 2151],
[0.2, 488, 1266],
[0.3, 367, 1069],
[0.4, 292, 925],
[0.5, 250, 829],
[0.6, 233, 813],
[0.7, 216, 739],
[0.8, 210, 727],
[0.9, 203, 699],
[1.0, 201, 709]
]

data = np.array(d0)

fig = plt.figure()
fig.subplots_adjust(hspace=0.7)
ax = axisartist.Subplot(fig, 111)
fig.add_subplot(ax)

ax.set_xscale("log")
ax.set_xticks(data[:,0])
ax.set_xticklabels([str(i) for i in data[:,0]])

ax.set_yscale("log")
yticks1 = [int2int(i) for i in data[:,1]]
yticks2 = [int2int(i) for i in data[:,2]]
yticks = np.unique(yticks1 + yticks2)
ax.set_yticks(yticks)
ax.set_yticklabels([str(int(i)) for i in yticks])

ax.plot(data[:,0], data[:,1], '-^', color='red', linewidth=1.5, label='MAE')
ax.plot(data[:,0], data[:,2], '-x', color='blue', linewidth=1.5, label='RMSE')

ax.legend()
ax.set_ylabel('[mm]')
ax.set_xlabel('Keep ratio (%)')
# ax.set_xlabel('Number of scan lines')

ax.grid()
plt.show()
