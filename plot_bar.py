import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

d0=[\
[1, 288, 927],
[2, 226, 789],
[3, 241, 866],
[4, 210, 716],
]

data = np.array(d0)

fig = plt.figure()
fig.subplots_adjust(hspace=0.7)
ax = axisartist.Subplot(fig, 111)
fig.add_subplot(ax)

ax.set_xticks(data[:,0])
ax.set_xticklabels(['({})'.format(i) for i in data[:,0]])

ax.bar(data[:,0]-0.2, data[:,1], alpha=0.5, width=0.4, color='blue', label='MAE', lw=0)
ax.bar(data[:,0]+0.2, data[:,2], alpha=0.5, width=0.4, color='green', label='RMSE', lw=0)

for a, b in zip(data[:,0]-0.2, data[:,1]):
    plt.text(a, b + 0.1, '%.0f' % b, ha='center', va='bottom', fontsize=10)
for a, b in zip(data[:,0]+0.2, data[:,2]):
    plt.text(a, b + 0.1, '%.0f' % b, ha='center', va='bottom', fontsize=10)

ax.legend()
ax.set_ylabel('[mm]')

ax.grid(False)
plt.show()
