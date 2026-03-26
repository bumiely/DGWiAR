import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import scipy.io as sio
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.image import RecurrencePlot
import matplotlib.patches as mpatches

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

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
legend_patches = [mpatches.Patch(color=color, label=f'采样点 {point}') for point, color in zip(highlight_points, highlight_colors)]
legend = plt.legend(handles=legend_patches,fontsize=14)
colors = ['red','green']
for i, text in enumerate(legend.get_texts()):
    text.set_color(colors[i])
# 设置刻度值字体大小
plt.tick_params(axis='both', labelsize=15)  # 设置 x 轴和 y 轴刻度字体大小为 12
plt.savefig('.\\out\\rp_1.png', bbox_inches='tight', dpi=600)



#########
# paa_data=paa_data.reshape(224,)
# # Visualize transformer sequence
# plt.plot(paa_data,alpha=0.7)  # 绘制整体曲线
#
# plt.plot(range(184, 194), paa_data[184:194].reshape(10,), color='green', label='采样点 184-194')
# plt.plot(range(132, 138), paa_data[132:138].reshape(6,), color='red', label='采样点 132-138')
# plt.tick_params(axis='both', labelsize=15)  # 设置 x 轴和 y 轴刻度字体大小为 12
# plt.xlabel('时间（采样点）',fontsize=15)
# plt.ylabel('幅值',fontsize=15)
# plt.legend(loc='lower right', fontsize=14)
# plt.savefig(".\\out\\rp2.png", bbox_inches='tight', dpi=600)
# plt.show()