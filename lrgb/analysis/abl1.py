import matplotlib.pyplot as plt

layer = [3, 4, 5, 6, 7]

gcn = [1, 1, 1, 0.21, 0.03]
gcn_nba = [1, 1, 1, 0.99, 0.3]
gin = [1, 1, 1, 0.05, 0.03]
gin_nba = [1, 1, 1, 0.98, 0.3]

linewidth = 4.0
capsize=3
capthick=2
yaxis = [0.0, 0.25, 0.5, 0.75, 1.0]
blue=(100/256, 110/256, 240/256)
green=(90/256, 200/256, 150/256)
red=(233/256, 60/256, 40/256)

plt.figure(figsize=(6, 4.4))
plt.rc('font', size=12)        
plt.rc('axes', labelsize=24)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend', fontsize=8)
plt.xlabel('Number of layers')
ylabel = plt.ylabel('Train accuracy ↑')

plt.plot(layer, gcn, c=blue, label='GCN', linewidth=linewidth, linestyle='dotted')
plt.plot(layer, gcn_nba, c=blue, label='GCN+NBA', linewidth=linewidth)
plt.plot(layer, gin, c=green, label='GIN', linewidth=linewidth, linestyle='dotted')
plt.plot(layer, gin_nba, c=green, label='GIN+NBA', linewidth=linewidth)

plt.tight_layout()
plt.xticks(layer)
plt.yticks(yaxis)
plt.legend(fontsize="x-large")
plt.grid(True)
plt.show()

plt.savefig('./figure/abl1.png')