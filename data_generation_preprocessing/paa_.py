from matplotlib import pyplot as plt
from pyts.approximation import PiecewiseAggregateApproximation
import scipy.io as sio



mat_data = sio.loadmat("E:\\exper_datasave\\csi data\\work2\\20181109_60hz_10o_mat\\20181109_threeants\\5-3-1-5-1.mat")
data = mat_data['traindata']
num = data.shape[1]
sub = data[89,:].reshape(1, -1)
print(sub.shape)
transformer = PiecewiseAggregateApproximation(window_size=None,output_size=224,overlapping=False)
paa_data = transformer.transform(sub)
# 绘制图形
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# 在第一个子图上绘制原始时间序列
axes[0].plot(sub[0], color='blue')
axes[0].set_title('Original Time Series')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')

# 在第二个子图上绘制PAA处理后的时间序列
axes[1].plot(paa_data[0], color='red')
axes[1].set_title('PAA Time Series')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Value')

# 调整布局
plt.tight_layout()
plt.show()