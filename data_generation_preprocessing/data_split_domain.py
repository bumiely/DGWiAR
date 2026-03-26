import os

# no_domain_distinguish
"""
open()中w+:每次都会将原有的文件覆盖，如果没有的话就会创建并写入
Python中单引号和双引号都可以用来表示一个字符串
source.write(file_name+'\n')时加\n或者\r都能换行,如果加\r\n则是中间空了一行
"""
# def ganlist_simplefolder():
#     root_dir = 'E:/exper_datasave/csi data/CSItoImage'
#     sub_dir = "temp/user1-2-3-p"
#     target_path = os.path.join(root_dir, 'target_data.txt')
#     source_path = os.path.join(root_dir, 'source_data.txt')
#
#     with open(target_path, 'w+') as target, open(source_path, 'w+') as source:
#         train_dir = os.path.join(root_dir, sub_dir)
#         img_files = os.listdir(train_dir)
#
#         for file_name in img_files:
#             source.write(file_name+'\n')

# domain_distinguish
def ganlist_domain_split():
    root_dir = "E:\\exper_datasave\\csi data\\work2\\20181109_imageset_2"
    sub_dir = "user123\\"
    target_path = os.path.join(root_dir, 'target_data.txt')
    source_path = os.path.join(root_dir, 'source_data.txt')

    with open(target_path, 'w+') as target, open(source_path, 'w+') as source:
        train_dir = os.path.join(root_dir, sub_dir)
        img_files = os.listdir(train_dir)

        for file_name in img_files:
            parts = file_name.split('-')  # 1-1-1-1-1.png-->['user1', '1', '1', '1', '1', 'r6.png']
            if parts[2] == '1':
                target.write(file_name + '\n')
            else:
                source.write(file_name + '\n')

if __name__=='__main__':

    ganlist_domain_split()