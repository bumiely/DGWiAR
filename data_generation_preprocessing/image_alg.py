import numpy as np
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.datasets import load_gunpoint
from pyts.image import RecurrencePlot, GramianAngularField,MarkovTransitionField
import os
import scipy.io as sio
from matplotlib import pyplot as plt, image
from PIL import Image


mat_data = sio.loadmat("E:\\exper_datasave\\csi data\\work2\\20181109_60hz_10o_mat\\20181109_threeants\\5-3-1-5-1.mat")
data = mat_data['traindata']
num = data.shape[1]
sub = data[89,:].reshape(1, -1)
transformer = PiecewiseAggregateApproximation(window_size=data.shape[1]//224)
paa_data = transformer.transform(sub)
rp = RecurrencePlot(dimension=1, time_delay=1)
X_rp = rp.fit_transform(paa_data)
print(type(X_rp))  # 检查 sub1_rp 的类型
print(X_rp.shape)   # 检查 sub1_rp 的维度
# 获取图像数据并去掉批次维度
image_data = X_rp[0]

# 如果数据是浮动的，可能需要转换为 8 位整数类型
image_data = np.uint8(image_data)
image.imsave("rp.png", image_data)