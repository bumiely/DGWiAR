import numpy as np
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.datasets import load_gunpoint
from pyts.image import RecurrencePlot, GramianAngularField,MarkovTransitionField
import os
import scipy.io as sio
from matplotlib import pyplot as plt, image
from PIL import Image

# # Recurrence Plots
# path = 'E:/工作台存储/csi资料存储/CSI3_WIDAR/data/'
#
# img_path = os.listdir(path)
# for line in img_path:
#         mat_data = sio.loadmat(path + line)
#         data = mat_data['traindata']
#         num = data.shape[1]
#         for k in range(num):
#             sub= data[k,:].reshape(1, -1)
#             transformer = PiecewiseAggregateApproximation(window_size=data.shape[1]//224)
#             paa_data = transformer.transform(sub)
#             test = data[::5, k].reshape(1, -1)
#             rp = RecurrencePlot(dimension=1, time_delay=1)
#             X_rp = rp.fit_transform(test)
#             imagename = f"E:/exper_datasave/csi data/RT/" +line.split('.')[0] +'-'+str(count)+ ".png"
#             image.imsave(imagename, X_rp[0])
#             count = count + 1



# GAF
# from pyts.image import GramianAngularField
# path4='E:/工作台存储/csi资料存储/CSI3_WIDAR/data/'
# img_path=os.listdir(path4)
# for line in img_path:
#         data = sio.loadmat('E:\工作台存储\csi资料存储\CSI3_WIDAR\data/'+line)
#
#         train_data = data['traindata'] #  shape=1344x30
#         img_num = train_data.shape[1]
#         img_size =1120
#
#         count = 1
#         for k in range(img_num):
#             # test = train_data[::5, k].reshape(1, -1)
#             test = train_data[:, k].reshape(1, -1)
#             gaf = GramianAngularField(image_size=224, sample_range=(0, 1), method='summation')
#             img_gaf = gaf.fit_transform(test)
#             imagename = f"E:/exper_datasave/csi data/gaf31/data/" +line.split('.')[0] +'-'+str(count)+ ".png"
#             image.imsave(imagename, img_gaf[0])
#             count = count + 1

#####################################


if __name__ == '__main__':
    from cross_ants_subtrans import process_and_save_images

    mat_data = sio.loadmat('E:\\exper_datasave\\csi data\\work2\\20181109_60hz_10o_mat\\20181109_threeants\\5-3-1-5-1.mat')
    data = mat_data['traindata']
    save_path = 'E:/exper_datasave/csi data/work2/20181109_60hz_10o_mat/temp/'
    save_concat_path = 'E:/exper_datasave/csi data/work2/20181109_60hz_10o_mat/temp/'

    process_and_save_images(data, 25, save_path, save_concat_path,"1-11-1-1.mat")






















