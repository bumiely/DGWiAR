import os

import numpy as np
import scipy.io as sio
import PIL.Image as Image
from matplotlib import image, pyplot as plt
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.image import RecurrencePlot, GramianAngularField, MarkovTransitionField


# 定义函数
def extract_data(data, sub, interval=30):
    # 确保数据的shape符合你的期望
    return [data[sub + i * interval, :].reshape(1, -1) for i in range(3)]

def RP_image(sub):
    # 使用PAA进行降维
    transformer = PiecewiseAggregateApproximation(window_size=None, output_size=224, overlapping=False)
    sub = transformer.fit_transform(sub)
    # 使用RecurrencePlot进行变换
    rp = RecurrencePlot()
    sub_rp = rp.fit_transform(sub)
    return sub_rp

def gaf_s(sub):
    # 使用Gramian Angular Field进行变换
    gaf_s = GramianAngularField(image_size=224, sample_range=(0, 1), method='summation')
    sub_gaf_s = gaf_s.fit_transform(sub)
    return sub_gaf_s

def gaf_d(sub):
    # 使用Gramian Angular Field进行变换，方法为空字符串 ''
    gaf_d = GramianAngularField(image_size=224, sample_range=(0, 1), method='difference')
    sub_gaf_d = gaf_d.fit_transform(sub)
    return sub_gaf_d

def mtf(sub):
    # 使用Markov Transition Field进行变换
    mtf = MarkovTransitionField(image_size=224, n_bins=8)
    sub_m = mtf.fit_transform(sub)
    return sub_m

# 读取并拼接图像的函数
def concat_images(sub_id,temp_path):
    # 根据sub_n来动态生成文件名
    rp = image.imread(os.path.join(temp_path, f'sub{sub_id}_rp.png'))
    gs = image.imread(os.path.join(temp_path, f'sub{sub_id}_gs.png'))
    gd = image.imread(os.path.join(temp_path, f'sub{sub_id}_gd.png'))
    m = image.imread(os.path.join(temp_path, f'sub{sub_id}_m.png'))

    # 横向拼接：将四张图像横向拼接
    horizontal_concat = np.concatenate((rp, gs, gd, m), axis=1)

    return horizontal_concat


def process_and_save_images(data, sub_n, temp_path, save_path,line):
    """
    :param data: 输入的数据，应该是一个包含时序数据的矩阵
    :param sub_n: 子集数据的索引值
    :param temp_path: 图像保存路径
    :param save_concat_path: 拼接图像保存路径
    """
    # 提取数据
    sub1, sub2, sub3 = extract_data(data, sub_n)

    # 处理每个子集的PAA和RecurrencePlot
    sub1_rp = RP_image(sub1)
    sub2_rp = RP_image(sub2)
    sub3_rp = RP_image(sub3)

    # 使用Gramian Angular Field生成转换后的图像
    sub1_gs = gaf_s(sub1)
    sub2_gs = gaf_s(sub2)
    sub3_gs = gaf_s(sub3)

    # 使用Gramian Angular Field生成转换后的图像
    sub1_gd = gaf_d(sub1)
    sub2_gd = gaf_d(sub2)
    sub3_gd = gaf_d(sub3)

    # 使用Markov Transition Field生成转换后的图像
    sub1_m = mtf(sub1)
    sub2_m = mtf(sub2)
    sub3_m = mtf(sub3)

    # 保存结果
    image.imsave(os.path.join(temp_path, f'sub1_rp.png'), sub1_rp[0])
    image.imsave(os.path.join(temp_path, f'sub2_rp.png'), sub2_rp[0])
    image.imsave(os.path.join(temp_path, f'sub3_rp.png'), sub3_rp[0])

    image.imsave(os.path.join(temp_path, f'sub1_gs.png'), sub1_gs[0])
    image.imsave(os.path.join(temp_path, f'sub2_gs.png'), sub2_gs[0])
    image.imsave(os.path.join(temp_path, f'sub3_gs.png'), sub3_gs[0])

    image.imsave(os.path.join(temp_path, f'sub1_gd.png'), sub1_gd[0])
    image.imsave(os.path.join(temp_path, f'sub2_gd.png'), sub2_gd[0])
    image.imsave(os.path.join(temp_path, f'sub3_gd.png'), sub3_gd[0])

    image.imsave(os.path.join(temp_path, f'sub1_m.png'), sub1_m[0])
    image.imsave(os.path.join(temp_path, f'sub2_m.png'), sub2_m[0])
    image.imsave(os.path.join(temp_path, f'sub3_m.png'), sub3_m[0])

    # 横向拼接：
    sub1_concat = concat_images(1, temp_path)
    sub2_concat = concat_images(2, temp_path)
    sub3_concat = concat_images(3, temp_path)

    # 纵向拼接：
    final_concat = np.concatenate((sub1_concat, sub2_concat, sub3_concat), axis=0)

    # 提取line中的前缀部分
    file_prefix = line.split('.')[0]

    # 保存最终的拼接图像，文件名为line中的前缀部分加上.png
    final_concat_path = os.path.join(save_path, f'{file_prefix}.png')
    image.imsave(final_concat_path, final_concat)

    # 使用PIL读取保存的图像
    img = Image.open(final_concat_path)
    # 调整图像大小为224x224
    img_resized = img.resize((224, 224))
    # 再次保存调整大小后的图像到原路径
    img_resized.save(final_concat_path)


    # 删除原始图片
    # for sub_id in range(1, 4):
    #     for suffix in ['rp', 'gs', 'gd', 'm']:
    #         file_path = os.path.join(temp_path, f'sub{sub_id}_{suffix}.png')
    #         if os.path.exists(file_path):
    #             os.remove(file_path)


if __name__ == '__main__':

    path = 'E:\\exper_datasave\\csi data\\work2\\20181109_60hz_10o_mat\\20181109_threeants\\'
    img_path = os.listdir(path)
    temp_path = 'E:/exper_datasave/csi data/work2/20181109_60hz_10o_mat/temp/'
    save_path = 'E:\\exper_datasave\\csi data\\work2\\20181109_imageset_1\\'

    for line in img_path:

        mat_data = sio.loadmat(path + line)
        data = mat_data['traindata']
        process_and_save_images(data, 25, temp_path, save_path, line)
        print(f"正在处理{line}")
    print("全部处理完毕")












