# 可视化 sub1, sub2, sub3
from matplotlib import pyplot as plt
import scipy.io as sio



mat_data = sio.loadmat("E:\\exper_datasave\\csi data\\work2\\20181109_60hz_10o_mat\\20181109_threeants\\1-1-1-1-1.mat")
data = mat_data['traindata']
# fix = 0
# sub1 = data[fix, :].reshape(1, -1)
# sub2 = data[fix+30, :].reshape(1, -1)
# sub3 = data[fix+60, :].reshape(1, -1)
def extract_data(sub, interval=30):
    return [data[sub + i * interval, :].reshape(1, -1) for i in range(3)]
sub1,sub2,sub3 = extract_data(0)
plt.figure(figsize=(10, 6))
# 绘制 sub1
plt.plot(sub1.flatten(), label='Sub1', color='b')
# 绘制 sub2
plt.plot(sub2.flatten(), label='Sub2', color='g')
# 绘制 sub3
plt.plot(sub3.flatten(), label='Sub3', color='r')
# 添加标题和标签
plt.title("subcarrier index = 17")
plt.xlabel("Index")
plt.ylabel("Value")
# 显示图例
plt.legend()
# 显示图像
plt.show()