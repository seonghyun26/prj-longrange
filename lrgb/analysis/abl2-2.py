import matplotlib.pyplot as plt

layer = [5, 10, 15, 20]

gcn_bb = [0.7099, 0.7207, 0.7175, 0.7140]
gcn_bb_std = [0.0010, 0.0028, 0.0050, 0.0031]
gcn_nba = [0.6977, 0.7015, 0.7000, 0.6986]
gcn_nba_std = [0.0010, 0.0009, 0.0066, 0.0050]
gcn_ba = [0.7067, 0.7032, 0.7038, 0.6953]
gcn_ba_std = [0.0045, 0.0034, 0.0005, 0.0134]
gcn_vnl = [0.6833, 0.6800, 0.6762, 0.6697]
gcn_vnl_std = [0.0037, 0.0015, 0.0090, 0.0020]

labelFontSize=15
elinewidth = 2.0
linewidth = 4.0
capsize = 3
capthick = 2
yaxis = [0.64, 0.67, 0.70, 0.73]
blue=(100/256, 110/256, 240/256)
green=(90/256, 200/256, 150/256)
red=(233/256, 60/256, 40/256)
purple=(200/256, 100/256, 200/256)

plt.figure(figsize=(6, 4.4))
plt.rc('font', size=10)
plt.rc('axes', labelsize=24)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend', fontsize=10)
plt.xlabel('Number of layers')
plt.ylabel('AP ↑')


plt.plot(layer, gcn_bb, c=blue, label='NBA-GCN w/ BG', linewidth=linewidth)
plt.errorbar(
    layer,
    gcn_bb,
    gcn_bb_std,
    c=blue,
    elinewidth=elinewidth,
    capsize=capsize, capthick=capthick, linewidth=linewidth
)

# plt.plot(layer, gcn_nba, c=blue, label='NBA-GCN', linewidth=linewidth)
# plt.errorbar(
#     layer,
#     gcn_nba,
#     gcn_nba_std,
#     c=blue,
#     elinewidth=elinewidth,
#     capsize=capsize, capthick=capthick, linewidth=linewidth
# )

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
    elinewidth=elinewidth,
    capsize=capsize, capthick=capthick, linewidth=linewidth
)

plt.tight_layout()
plt.xticks(layer)
plt.yticks(yaxis)
plt.legend(fontsize="x-large")
plt.grid(True)
plt.annotate(
    f'{gcn_bb[1]}',
    (10, gcn_bb[1]),
    textcoords="offset points",
    xytext=(-4,6),
    ha='right',
    color=blue,
    weight='bold',
    fontsize=labelFontSize
)
# plt.annotate(
#     f'{gcn_nba[1]}',
#     (10, gcn_nba[1]),
#     textcoords="offset points",
#     xytext=(-4,20),
#     ha='right',
#     color=blue,
#     weight='bold',
#     fontsize=labelFontSize
# )
plt.annotate(
    f'{gcn_ba[0]}',
    (5, gcn_ba[0]),
    textcoords="offset points",
    xytext=(6,-30),
    ha='left',
    color=green,
    weight='bold',
    fontsize=labelFontSize
)
plt.annotate(
    f'{gcn_vnl[0]}',
    (5, gcn_vnl[0]),
    textcoords="offset points",
    xytext=(6,-30),
    ha='left',
    color=red,
    weight='bold',
    fontsize=labelFontSize
)
plt.show()

plt.savefig('./figure/abl2-2.pdf')