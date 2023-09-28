import matplotlib.pyplot as plt

layer = [6, 8, 10, 12]

gcn_nba = [0.2877, 0.2907,	0.2963,	0.3009]
gcn_nba_std = [0.0003, 0.0034, 0.0002, 0.0006]
gcn_ba = [0.24,	0.2421,	0.2587,	0.2591]
gcn_ba_std = [0.0010, 0.0057, 0.0042, 0.0043]
# gcn_vnl = [0.1118, 0.1098, 0.1091, 0.1044]
# gcn_vnl_std = [0.0054, 0.0072, 0.0045, 0.0027]
gcn_vnl = [0.2045, 0.2068, 0.2057, 0.2050]
gcn_vnl_std = [0.0029, 0.0027, 0.0037, 0.0030]

labelFontSize = 15
linewidth = 4.0
capsize=3
capthick=2
yaxis = [0.12, 0.16, 0.20, 0.24, 0.28, 0.32]
blue=(100/256, 110/256, 240/256)
green=(90/256, 200/256, 150/256)
red=(233/256, 60/256, 40/256)

plt.figure(figsize=(6, 4.4))
plt.rc('font', size=10)
plt.rc('axes', labelsize=24)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend', fontsize=10)
plt.xlabel('Number of layers')
plt.ylabel('F1 score ↑')


plt.plot(layer, gcn_nba, c=blue, label='NBA-GCN', linewidth=linewidth)
plt.errorbar(
    layer,
    gcn_nba,
    gcn_nba_std,
    c=blue,
    elinewidth=2,
    capsize=capsize, capthick=capthick, linewidth=linewidth
)

plt.plot(layer, gcn_ba, c=green, label='BA-GCN', linewidth=linewidth)
plt.errorbar(
    layer,
    gcn_ba,
    gcn_ba_std,
    c=green,
    elinewidth=2,
    capsize=capsize, capthick=capthick, linewidth=linewidth
)

plt.plot(layer, gcn_vnl, c=red, label='GCN', linewidth=linewidth)
plt.errorbar(
    layer,
    gcn_vnl,
    gcn_vnl_std,
    c=red,
    elinewidth=2,
    capsize=capsize, capthick=capthick, linewidth=linewidth
)

plt.tight_layout()
plt.xticks(layer)
plt.yticks(yaxis)
plt.legend(fontsize="x-large", loc='lower left')
plt.grid(True)
plt.annotate(
    f'{gcn_nba[3]}',
    (12, gcn_nba[3]),
    textcoords="offset points",
    xytext=(-4,6),
    ha='right',
    color=blue,
    weight='bold',
    fontsize=labelFontSize
)
plt.annotate(
    f'{gcn_ba[3]}',
    (12, gcn_ba[3]),
    textcoords="offset points",
    xytext=(-4,8),
    ha='right',
    color=green,
    weight='bold',
    fontsize=labelFontSize
)
plt.annotate(
    f'{gcn_vnl[1]}',
    (8, gcn_vnl[1]),
    textcoords="offset points",
    xytext=(-4, 8),
    ha='right',
    color=red,
    weight='bold',
    fontsize=labelFontSize
)
plt.show()

plt.savefig('./figure/abl2-1.pdf')