import torch
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.image import RecurrencePlot
import matplotlib.patches as mpatches
plt.rcParams['font.family']='Times New Roman, SimHei'
data = sio.loadmat('E:/exper_datasave/csi data/20181109_3ants/data/4-1-1-1-1.mat')
csi_data = data['traindata']  # (90, 1344)
x=csi_data[74,:]
x=x.reshape(1,1344)
print(x.shape)
# img_num = x.shape[0]
transformer = PiecewiseAggregateApproximation(window_size=x.shape[1]//224)
paa_data = transformer.transform(x)
# print(paa_data.shape)
rp = RecurrencePlot(dimension=1, time_delay=1)
rp_data = rp.fit_transform(paa_data)
# print(rp_data.shape)
# image.imsave('E:/组会/PPT/picture/rp.png', rp_data[0])

plt.figure(figsize=(8, 6))
############
# 显示图像
plt.imshow(rp_data[0], cmap='viridis', origin='lower')
# 突出显示多个点并获取它们的值
highlight_points = [(132, 138), (184, 194)]  # 注意这里的坐标是(列, 行)
highlight_values = [rp_data[0][point[::-1]] for point in highlight_points]

# 获取颜色映射中特定值对应的颜色
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(vmin=np.min(rp_data[0]), vmax=np.max(rp_data[0]))
highlight_colors = [cmap(norm(value)) for value in highlight_values]

# 创建虚拟样本，并添加到图例中
legend_patches = [mpatches.Patch(color=color, label=f'Point {point}') for point, color in zip(highlight_points, highlight_colors)]
legend = plt.legend(handles=legend_patches,fontsize=18)
colors=['red','C1']
for i, text in enumerate(legend.get_texts()):
    text.set_color(colors[i])
plt.tick_params(axis='both', labelsize=18)
plt.xlim(0, 224)
plt.xticks(np.arange(0, 224,50))
plt.savefig('./out/rp2.png', bbox_inches='tight',dpi=300)
plt.show()


#########
# paa_data=paa_data.reshape(224,)
# # Visualize transformer sequence
# plt.figure(figsize=(10, 6))
# plt.plot(paa_data)  # 绘制整体曲线
# plt.plot(range(184, 194), paa_data[184:194].reshape(10,), color='green', label='时间点 184~194 幅值变化')
# plt.plot(range(132, 138), paa_data[132:138].reshape(6,), color='red', label='时间点 132~138 幅值变化')
# plt.tick_params(axis='both', labelsize=25)
# plt.ylim(0, 9)
# plt.yticks(np.arange(0, 9, 1))
# plt.xlabel('采样时间点',fontsize=25)
# plt.ylabel('幅值',fontsize=25)
# plt.legend(loc='lower right',fontsize=25)
# plt.savefig('./out/rp1.svg', bbox_inches='tight',dpi=300)
# plt.show()