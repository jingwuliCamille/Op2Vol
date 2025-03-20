import torch
import numpy as np
import cv2
from PIL import Image
import os
import re

from torch.utils.data.dataset import Dataset
try:
    from sklearn.externals import joblib
except:
    import joblib

def find_most_white_png(current_directory):
    max_white_pixel_count = 0
    max_white_pixel_image = None

    # 遍历当前文件夹下所有的 .png 文件
    for file in os.listdir(current_directory):
        if file.lower().endswith('.png'):
            file_path = os.path.join(current_directory, file)

            try:
                # 尝试读取 PNG 文件
                image = Image.open(file_path)

                # 如果图像是灰度图像 'L'，则处理单通道像素值
                if image.mode == 'L':
                    white_pixel_count = sum(1 for pixel in image.getdata() if pixel == 255)
                else:
                    # 其他模式的处理，比如 'RGB'
                    white_pixel_count = sum(1 for pixel in image.getdata() if pixel == (255, 255, 255))

                # 判断是否是当前最大的白色像素数量
                if white_pixel_count > max_white_pixel_count:
                    max_white_pixel_count = white_pixel_count
                    max_white_pixel_image = file
            except Exception as e:
                # 处理无法读取的文件
                print(f"无法读取文件 {file_path}: {e}")

    return max_white_pixel_image

class LIDC_IDRI_flow(Dataset):

    def __init__(self, root=None, load=False, training=True):
        self.ap = r'/raid/Data/lijingwu/'
       # self.ap_data = r'/home/zhangwei/optical_flow/'
        if root is None:
            root = os.path.join(self.ap, 'data_optical/')
        print("Loading file ", root)
        print('This data is for flow ' + ('training' if training else 'test'))
        self.root = root
        self.initIndex=[]
        self.load = load
        self.training = training
        print('split dataset')
        self.split_dataset()
    
    def split_dataset(self):
        if self.training:
            root = self.root
            self.data = dict() # { noduleId: path }
            files = os.listdir(root)
            nodule_id = 0
            for file in files:
                nodules = os.listdir(os.path.join(root, file))
                for nodule in nodules:
                    self.data[nodule_id] = os.path.join(root, file, nodule)
                    nodule_id += 1
            nodule_ids = list(range(len(self.data)))
            print("len(self.data)",len(self.data))
            train_num = int(len(self.data)*0.9)
            np.random.shuffle(nodule_ids)
            train_ids = np.random.choice(nodule_ids, size=train_num, replace=False)
           
            self.train = [] # [((img1, img2), (path, None)), ((img2, img3), (None, None))]
            
            self.train_set = [] # [ path1, path2, path3, ... ]
            
            for i in train_ids:
                path = self.data[i]
                images = os.listdir(os.path.join(path, 'images'))
                images.sort(key = self.my_sort)
                forward_images = tuple(os.path.join(path, 'images', image) for image in images)
                self.train.append(forward_images)
            print("self.train_set",len(self.train_set))
            print('[INFO]The length of train: ', len(self.train))
        else:
            root = self.root
            self.testdata = dict() # { noduleId: path }
            files = os.listdir(root)
            self.test = [] # [(((img1, img2, ..., img-/2), (mask1, mask2, ...)), (...))]
            test_set = [] # [ path1, path2, path3, ... ]
        
            print("files",files)
            for file in files:

                nodules = os.path.join(root, file)
            
              
                images = sorted(os.listdir(os.path.join(nodules, 'nodule-liver','images')), key=lambda x: int(re.search(r'\d+', x).group()))

                masks = sorted(os.listdir(os.path.join(nodules, 'nodule-liver','mask-0')), key=lambda x: int(re.search(r'\d+', x).group()))
                print(os.path.join(nodules, 'nodule-liver','mask-0'))
                init_file=find_most_white_png(os.path.join(nodules, 'nodule-liver','mask-0'))
                print("init_file",init_file)
                
                # print("init_file",init_file)
                if init_file:
                    match = re.search(r'\d+', init_file)
                
                    if match:
                        init_idx = int(match.group())
                        print(init_idx)

                        print(init_file)

                        # 找到 init_file 在 images 列表中的索引
                        init_idx = masks.index(init_file)

                        if len(images) < 2:
                            continue

                        init_idx = init_idx
                        backward_images = tuple([os.path.join(nodules, 'nodule-liver','images', image) for image in reversed(images[:init_idx + 1])])
                        backward_masks = tuple([os.path.join(nodules, 'nodule-liver','mask-0', mask) for mask in reversed(masks[:init_idx + 1])])
                        forward_images = tuple([os.path.join(nodules, 'nodule-liver','images', image) for image in images[init_idx:]])
                        forward_masks = tuple([os.path.join(nodules, 'nodule-liver','mask-0', mask) for mask in masks[init_idx:]])
                        self.test.append(
                            ((backward_images, backward_masks), (forward_images, forward_masks))
                        )
    def my_sort(self, strs):
        return int(strs.split('.')[0].split('-')[1])

    def process_mask(self, mask):
        mask[mask>=127.5] = 255
        mask[mask<127.5] = 0
        mask = mask
        return mask
    def pack_tensor(self, path, is_mask=False):
        path = path.replace('\\', '/').split('/')
        self.root = self.root.replace('\\','/')
        path = os.path.join(self.root, '/'.join(path[-4:]))
        print(path)
        
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = torch.unsqueeze(torch.from_numpy(image), 0).type(torch.FloatTensor)

        return image
    def __getitem__(self, index):
        if self.training:
            forward = self.train[index]
            forward_images = []

            for i in range(len(forward)):
                forward_images.append(self.pack_tensor(forward[i], is_mask=False))
            forward_images = torch.cat(forward_images, dim=0)
            print("forward_images.shape",forward_images.shape)
            return forward_images
        
    
        else:
            backward, forward = self.test[index]
            backward_images = []
            backward_masks = []
            assert len(backward[0]) == len(backward[1])
            for i in range(len(backward[0])):
                backward_images.append(self.pack_tensor(backward[0][i], is_mask=False))
                backward_masks.append(self.pack_tensor(backward[1][i], is_mask=True))
            backward_images = torch.cat(backward_images, dim=0)
            backward_masks = torch.cat(backward_masks, dim=0)
            forward_images = []
            forward_masks = []
            assert len(forward[0]) == len(forward[1])
            for i in range(len(forward[0])):
                forward_images.append(self.pack_tensor(forward[0][i], is_mask=False))
                forward_masks.append(self.pack_tensor(forward[1][i], is_mask=True))
            forward_images = torch.cat(forward_images, dim=0)
            forward_masks = torch.cat(forward_masks, dim=0)
            
            print("backward_images",backward_images.shape)
            print("backward_masks",backward_masks.shape)

            assert backward_images.shape == backward_masks.shape
            assert forward_images.shape == forward_masks.shape
            
            return backward_images, backward_masks, forward_images, forward_masks


    # Override to give PyTorch size of dataset
    def __len__(self):
        if self.training:
            return len(self.train)
        else:
            return len(self.test)

class Chaos(Dataset):

    def __init__(self, root=None, load=False, training=True):
        self.ap = r'/raid/Data/lijingwu/'
       # self.ap_data = r'/home/zhangwei/optical_flow/'
        if root is None:
            root = os.path.join(self.ap, 'data_optical/')
        print("Loading file ", root)
        print('This data is for flow ' + ('training' if training else 'test'))
        self.root = root
        self.initIndex=[]
        self.load = load
        self.training = training
        print('split dataset')
        self.split_dataset()
    
    def __len__(self):
            if self.training:
                return len(self.train)
            else:
                return len(self.test)
    def split_dataset(self):
        if self.training:
            root = self.root
            self.data = dict() # { noduleId: path }
            files = os.listdir(root)
            nodule_id = 0
            for file in files:
                nodules = os.listdir(os.path.join(root, file))
                for nodule in nodules:
                    self.data[nodule_id] = os.path.join(root, file, nodule)
                    nodule_id += 1
            nodule_ids = list(range(len(self.data)))
            train_num = int(len(self.data)*0.1)
            #test_num = len(self.data) - train_num
            np.random.shuffle(nodule_ids)
            train_ids = np.random.choice(nodule_ids, size=train_num, replace=False)
            self.train = [] # [((img1, img2), (path, None)), ((img2, img3), (None, None))]
            
            train_set = [] # [ path1, path2, path3, ... ]
            
            for i in nodule_ids:
                path = self.data[i]
                images = os.listdir(os.path.join(path, 'images'))
                #masks = os.listdir(os.path.join(path, 'mask-0'))
                images.sort(key = self.my_sort)
                #masks.sort(key = self.my_sort)
                if len(images) < 2:
                    continue
                if i in train_ids:
                    #self.train += self.data[i]
                    train_set.append(path.replace('\\', '/'))
                    mask_idx = [len(images) // 2]
                    for j in range(len(images) - 1):
                        path1 = os.path.join(path, 'images', 'slice-' + str(j) + '.png')
                        path2 = os.path.join(path, 'images', 'slice-' + str(j + 1) + '.png')
                        #print(path2)
                        #mask_flag1 = os.path.join(path, 'mask-0', 
                                                #'slice-' + str(j) + '.png') if j in mask_idx else None
                        #mask_flag2 = os.path.join(path, 'mask-0',
                                                #'slice-' + str(j + 1) + '.png') if (j + 1) in mask_idx else None
                        self.train += [
                                ((path1, path2)),#中间掩码
                                ((path2, path1))
                            ] 
            print('[INFO]The length of train: ', len(self.train))
        else:
            root = self.root
            self.testdata = dict() # { noduleId: path }
            files = os.listdir(root)
            self.test = [] # [(((img1, img2, ..., img-/2), (mask1, mask2, ...)), (...))]
            test_set = [] # [ path1, path2, path3, ... ]
        
            print("files",files)
            for file in files:

                nodules = os.path.join(root, file)
            
              
                images = sorted(os.listdir(os.path.join(nodules, 'image1')), key=lambda x: int(re.search(r'\d+', x).group()))

                masks = sorted(os.listdir(os.path.join(nodules, 'Ground')), key=lambda x: int(re.search(r'\d+', x).group()))
                print(os.path.join(nodules, 'Ground'))
                init_file=find_most_white_png(os.path.join(nodules, 'Ground'))
                print("init_file",init_file)
                
                # print("init_file",init_file)
                if init_file:
                    match = re.search(r'\d+', init_file)
                
                    if match:
                        init_idx = int(match.group())
                        print(init_idx)

                        print(init_file)

                        # 找到 init_file 在 images 列表中的索引
                        init_idx = masks.index(init_file)

                        if len(images) < 2:
                            continue

                        init_idx = init_idx+3
                        backward_images = tuple([os.path.join(nodules, 'image1', image) for image in reversed(images[:init_idx + 1])])
                        backward_masks = tuple([os.path.join(nodules, 'Ground', mask) for mask in reversed(masks[:init_idx + 1])])
                        forward_images = tuple([os.path.join(nodules, 'image1', image) for image in images[init_idx:]])
                        forward_masks = tuple([os.path.join(nodules, 'Ground', mask) for mask in masks[init_idx:]])
                        self.test.append(
                            ((backward_images, backward_masks), (forward_images, forward_masks))
                        )
                        # print("/n  backward_masks ",backward_masks)
                        # print("/n  forward_masks ",forward_masks)

    
    def pack_tensor(self, path, is_mask=False):
        path = path.replace('\\', '/').split('/')
        self.root = self.root.replace('\\','/')
        path = os.path.join(self.root, '/'.join(path[-3:]))
        
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
       

        image = torch.unsqueeze(torch.from_numpy(image), 0).type(torch.FloatTensor)

        return image

    def __getitem__(self, index):
        if self.training:
            images, mask_flags = self.train[index]
            a,b = images
            print('a',a)
            num = int(a.split('/')[9].split('-')[1].split('.')[0] )
            c,d = mask_flags
            #print(c,d)
            num_len = len(os.listdir(self.root + a.split('/')[6] + '/' +  a.split('/')[7] + '/' + a.split('/')[8]))
            imf = images[0]
            image1 = self.pack_tensor(images[0], is_mask=False)
            image2 = self.pack_tensor(images[1], is_mask=False)
            mask1 = self.pack_tensor(mask_flags[0],
                                     is_mask=True) if mask_flags[0] is not None else torch.ones_like(image1) * -1
            mask2 = self.pack_tensor(mask_flags[1],
                                     is_mask=True) if mask_flags[1] is not None else torch.ones_like(image2) * -1
            return image1, image2, mask1, mask2, num_len, num,imf
        else:
            backward, forward = self.test[index]
            backward_images = []
            backward_masks = []
            assert len(backward[0]) == len(backward[1])
            for i in range(len(backward[0])):
                backward_images.append(self.pack_tensor(backward[0][i], is_mask=False))
                backward_masks.append(self.pack_tensor(backward[1][i], is_mask=True))
            backward_images = torch.cat(backward_images, dim=0)
            backward_masks = torch.cat(backward_masks, dim=0)
            forward_images = []
            forward_masks = []
            assert len(forward[0]) == len(forward[1])
            for i in range(len(forward[0])):
                forward_images.append(self.pack_tensor(forward[0][i], is_mask=False))
                forward_masks.append(self.pack_tensor(forward[1][i], is_mask=True))
            forward_images = torch.cat(forward_images, dim=0)
            forward_masks = torch.cat(forward_masks, dim=0)
            
            print("backward_images",backward_images.shape)
            print("backward_masks",backward_masks.shape)

            assert backward_images.shape == backward_masks.shape
            assert forward_images.shape == forward_masks.shape
            
            return backward_images, backward_masks, forward_images, forward_masks
                
class Chaos1(Dataset):

    def __init__(self, root=None, load=False, training=True):
        self.ap = r'/raid/Data/lijingwu/'
       # self.ap_data = r'/home/zhangwei/optical_flow/'
        if root is None:
            root = os.path.join(self.ap, 'data_optical/')
        print("Loading file ", root)
        print('This data is for flow ' + ('training' if training else 'test'))
        self.root = root
        self.initIndex=[]
        self.load = load
        self.training = training
        print('split dataset')
        self.split_dataset()
    
    def __len__(self):
            if self.training:
                return len(self.train)
            else:
                return len(self.test)
    def split_dataset(self):
        if self.training:
            root = self.root
            self.data = dict() # { noduleId: path }
            files = os.listdir(root)
            nodule_id = 0
            for file in files:
                nodules = os.listdir(os.path.join(root, file))
                for nodule in nodules:
                    self.data[nodule_id] = os.path.join(root, file, nodule)
                    nodule_id += 1
            nodule_ids = list(range(len(self.data)))
            train_num = int(len(self.data)*0.1)
            #test_num = len(self.data) - train_num
            np.random.shuffle(nodule_ids)
            train_ids = np.random.choice(nodule_ids, size=train_num, replace=False)
            self.train = [] # [((img1, img2), (path, None)), ((img2, img3), (None, None))]
            
            train_set = [] # [ path1, path2, path3, ... ]
            
            for i in nodule_ids:
                path = self.data[i]
                images = os.listdir(os.path.join(path, 'images'))
                #masks = os.listdir(os.path.join(path, 'mask-0'))
                images.sort(key = self.my_sort)
                #masks.sort(key = self.my_sort)
                if len(images) < 2:
                    continue
                if i in train_ids:
                    #self.train += self.data[i]
                    train_set.append(path.replace('\\', '/'))
                    mask_idx = [len(images) // 2]
                    for j in range(len(images) - 1):
                        path1 = os.path.join(path, 'images', 'slice-' + str(j) + '.png')
                        path2 = os.path.join(path, 'images', 'slice-' + str(j + 1) + '.png')
                        #print(path2)
                        #mask_flag1 = os.path.join(path, 'mask-0', 
                                                #'slice-' + str(j) + '.png') if j in mask_idx else None
                        #mask_flag2 = os.path.join(path, 'mask-0',
                                                #'slice-' + str(j + 1) + '.png') if (j + 1) in mask_idx else None
                        self.train += [
                                ((path1, path2)),#中间掩码
                                ((path2, path1))
                            ] 
            print('[INFO]The length of train: ', len(self.train))
        else:
            root = self.root
            self.testdata = dict() # { noduleId: path }
            files = os.listdir(root)
            self.test = [] # [(((img1, img2, ..., img-/2), (mask1, mask2, ...)), (...))]
            test_set = [] # [ path1, path2, path3, ... ]
        
            print("files",files)
            for file in files:

                nodules = os.path.join(root, file)
            
              
                images = sorted(os.listdir(os.path.join(nodules, 'image')), key=lambda x: int(re.search(r'\d+', x).group()))

                masks = sorted(os.listdir(os.path.join(nodules, 'label')), key=lambda x: int(re.search(r'\d+', x).group()))
                print(os.path.join(nodules, 'label'))
                init_file=find_most_white_png(os.path.join(nodules, 'label'))
                print("init_file",init_file)
                
                # print("init_file",init_file)
                if init_file:
                    match = re.search(r'\d+', init_file)
                
                    if match:
                        init_idx = int(match.group())
                        print(init_idx)

                        print(init_file)

                        # 找到 init_file 在 images 列表中的索引
                        init_idx = masks.index(init_file)

                        if len(images) < 2:
                            continue

                        init_idx = init_idx+4
                        backward_images = tuple([os.path.join(nodules, 'image', image) for image in reversed(images[:init_idx + 1])])
                        backward_masks = tuple([os.path.join(nodules, 'label', mask) for mask in reversed(masks[:init_idx + 1])])
                        forward_images = tuple([os.path.join(nodules, 'image', image) for image in images[init_idx:]])
                        forward_masks = tuple([os.path.join(nodules, 'label', mask) for mask in masks[init_idx:]])
                        self.test.append(
                            ((backward_images, backward_masks), (forward_images, forward_masks))
                        )
                        # print("/n  backward_masks ",backward_masks)
                        # print("/n  forward_masks ",forward_masks)

    
    def pack_tensor(self, path, is_mask=False):
        path = path.replace('\\', '/').split('/')
        self.root = self.root.replace('\\','/')
        path = os.path.join(self.root, '/'.join(path[-3:]))
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
       

        image = torch.unsqueeze(torch.from_numpy(image), 0).type(torch.FloatTensor)

        return image

    def __getitem__(self, index):
        if self.training:
            images, mask_flags = self.train[index]
            a,b = images
            print('a',a)
            num = int(a.split('/')[9].split('-')[1].split('.')[0] )
            c,d = mask_flags
            #print(c,d)
            num_len = len(os.listdir(self.root + a.split('/')[6] + '/' +  a.split('/')[7] + '/' + a.split('/')[8]))
            imf = images[0]
            image1 = self.pack_tensor(images[0], is_mask=False)
            image2 = self.pack_tensor(images[1], is_mask=False)
            mask1 = self.pack_tensor(mask_flags[0],
                                     is_mask=True) if mask_flags[0] is not None else torch.ones_like(image1) * -1
            mask2 = self.pack_tensor(mask_flags[1],
                                     is_mask=True) if mask_flags[1] is not None else torch.ones_like(image2) * -1
            return image1, image2, mask1, mask2, num_len, num,imf
        else:
            backward, forward = self.test[index]
            backward_images = []
            backward_masks = []
            assert len(backward[0]) == len(backward[1])
            for i in range(len(backward[0])):
                backward_images.append(self.pack_tensor(backward[0][i], is_mask=False))
                backward_masks.append(self.pack_tensor(backward[1][i], is_mask=True))
            backward_images = torch.cat(backward_images, dim=0)
            backward_masks = torch.cat(backward_masks, dim=0)
            forward_images = []
            forward_masks = []
            assert len(forward[0]) == len(forward[1])
            for i in range(len(forward[0])):
                forward_images.append(self.pack_tensor(forward[0][i], is_mask=False))
                forward_masks.append(self.pack_tensor(forward[1][i], is_mask=True))
            forward_images = torch.cat(forward_images, dim=0)
            forward_masks = torch.cat(forward_masks, dim=0)
            
            print("backward_images",backward_images.shape)
            print("backward_masks",backward_masks.shape)

            assert backward_images.shape == backward_masks.shape
            assert forward_images.shape == forward_masks.shape
            
            return backward_images, backward_masks, forward_images, forward_masks

class Dircadb(Dataset):

    def __init__(self, root=None, load=False, training=True):
        self.ap = r'/raid/Data/lijingwu/'
       # self.ap_data = r'/home/zhangwei/optical_flow/'
        if root is None:
            root = os.path.join(self.ap, 'data_optical/')
        print("Loading file ", root)
        print('This data is for flow ' + ('training' if training else 'test'))
        self.root = root
        self.initIndex=[]
        self.load = load
        self.training = training
        print('split dataset')
        self.split_dataset()
    
    def __len__(self):
            if self.training:
                return len(self.train)
            else:
                return len(self.test)
    def split_dataset(self):
        if self.training:
            root = self.root
            self.data = dict() # { noduleId: path }
            files = os.listdir(root)
            nodule_id = 0
            for file in files:
                nodules = os.listdir(os.path.join(root, file))
                for nodule in nodules:
                    self.data[nodule_id] = os.path.join(root, file, nodule)
                    nodule_id += 1
            nodule_ids = list(range(len(self.data)))
            train_num = int(len(self.data)*0.1)
            #test_num = len(self.data) - train_num
            np.random.shuffle(nodule_ids)
            train_ids = np.random.choice(nodule_ids, size=train_num, replace=False)
            self.train = [] # [((img1, img2), (path, None)), ((img2, img3), (None, None))]
            
            train_set = [] # [ path1, path2, path3, ... ]
            
            for i in nodule_ids:
                path = self.data[i]
                images = os.listdir(os.path.join(path, 'images'))
                #masks = os.listdir(os.path.join(path, 'mask-0'))
                images.sort(key = self.my_sort)
                #masks.sort(key = self.my_sort)
                if len(images) < 2:
                    continue
                if i in train_ids:
                    #self.train += self.data[i]
                    train_set.append(path.replace('\\', '/'))
                    mask_idx = [len(images) // 2]
                    for j in range(len(images) - 1):
                        path1 = os.path.join(path, 'images', 'slice-' + str(j) + '.png')
                        path2 = os.path.join(path, 'images', 'slice-' + str(j + 1) + '.png')
                        #print(path2)
                        #mask_flag1 = os.path.join(path, 'mask-0', 
                                                #'slice-' + str(j) + '.png') if j in mask_idx else None
                        #mask_flag2 = os.path.join(path, 'mask-0',
                                                #'slice-' + str(j + 1) + '.png') if (j + 1) in mask_idx else None
                        self.train += [
                                ((path1, path2)),#中间掩码
                                ((path2, path1))
                            ] 
            print('[INFO]The length of train: ', len(self.train))
        else:
            root = self.root
            self.testdata = dict() # { noduleId: path }
            files = os.listdir(root)
            self.test = [] # [(((img1, img2, ..., img-/2), (mask1, mask2, ...)), (...))]
            test_set = [] # [ path1, path2, path3, ... ]
        
            print("files",files)
            for file in files:
                
                organs = os.listdir(os.path.join(root, file))
                print(organs)

                nodules = os.path.join(root, file, organs[0])
                images = sorted(os.listdir(os.path.join(nodules, 'images')), key=lambda x: int(re.search(r'\d+', x).group()))

                masks = sorted(os.listdir(os.path.join(nodules, 'mask-0')), key=lambda x: int(re.search(r'\d+', x).group()))
            
                init_file=find_most_white_png(os.path.join(nodules, 'mask-0'))
                print("init_file",init_file)
                
                # print("init_file",init_file)
                if init_file:
                    match = re.search(r'\d+', init_file)
                
                    if match:
                        init_idx = int(match.group())
                        print(init_idx)

                        print(init_file)

                        # 找到 init_file 在 images 列表中的索引
                        init_idx = masks.index(init_file)

                        if len(images) < 2:
                            continue

                        init_idx = init_idx+4
                        backward_images = tuple([os.path.join(nodules,'images', image) for image in reversed(images[:init_idx + 1])])
                        backward_masks = tuple([os.path.join(nodules, 'mask-0', mask) for mask in reversed(masks[:init_idx + 1])])
                        forward_images = tuple([os.path.join(nodules, 'images', image) for image in images[init_idx:]])
                        forward_masks = tuple([os.path.join(nodules, 'mask-0', mask) for mask in masks[init_idx:]])
                        self.test.append(
                            ((backward_images, backward_masks), (forward_images, forward_masks))
                        )
                        # print("/n  backward_masks ",backward_masks)
                        # print("/n  forward_masks ",forward_masks)

    def pack_tensor(self, path, is_mask=False):
        path = path.replace('\\', '/').split('/')
        self.root = self.root.replace('\\','/')
        path = os.path.join(self.root, '/'.join(path[-4:]))
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = torch.unsqueeze(torch.from_numpy(image), 0).type(torch.FloatTensor)

        return image

    def __getitem__(self, index):
        if self.training:
            images, mask_flags = self.train[index]
            a,b = images
            print('a',a)
            num = int(a.split('/')[9].split('-')[1].split('.')[0] )
            c,d = mask_flags
            #print(c,d)
            num_len = len(os.listdir(self.root + a.split('/')[6] + '/' +  a.split('/')[7] + '/' + a.split('/')[8]))
            imf = images[0]
            image1 = self.pack_tensor(images[0], is_mask=False)
            image2 = self.pack_tensor(images[1], is_mask=False)
            mask1 = self.pack_tensor(mask_flags[0],
                                     is_mask=True) if mask_flags[0] is not None else torch.ones_like(image1) * -1
            mask2 = self.pack_tensor(mask_flags[1],
                                     is_mask=True) if mask_flags[1] is not None else torch.ones_like(image2) * -1
            return image1, image2, mask1, mask2, num_len, num,imf
        else:
            backward, forward = self.test[index]
            backward_images = []
            backward_masks = []
            assert len(backward[0]) == len(backward[1])
            for i in range(len(backward[0])):
                backward_images.append(self.pack_tensor(backward[0][i], is_mask=False))
                backward_masks.append(self.pack_tensor(backward[1][i], is_mask=True))
            backward_images = torch.cat(backward_images, dim=0)
            backward_masks = torch.cat(backward_masks, dim=0)
            forward_images = []
            forward_masks = []
            assert len(forward[0]) == len(forward[1])
            for i in range(len(forward[0])):
                forward_images.append(self.pack_tensor(forward[0][i], is_mask=False))
                forward_masks.append(self.pack_tensor(forward[1][i], is_mask=True))
            forward_images = torch.cat(forward_images, dim=0)
            forward_masks = torch.cat(forward_masks, dim=0)
            
            print("backward_images",backward_images.shape)
            print("backward_masks",backward_masks.shape)

            assert backward_images.shape == backward_masks.shape
            assert forward_images.shape == forward_masks.shape
            
            return backward_images, backward_masks, forward_images, forward_masks



class Sliver(Dataset):

    def __init__(self, root=None, load=False, training=True):
        self.ap = r'/raid/Data/lijingwu/'
       # self.ap_data = r'/home/zhangwei/optical_flow/'
        if root is None:
            root = os.path.join(self.ap, 'data_optical/')
        print("Loading file ", root)
        print('This data is for flow ' + ('training' if training else 'test'))
        self.root = root
        self.initIndex=[]
        self.load = load
        self.training = training
        print('split dataset')
        self.split_dataset()
    
    def __len__(self):
            if self.training:
                return len(self.train)
            else:
                return len(self.test)
    def split_dataset(self):
        if self.training:
            root = self.root
            self.data = dict() # { noduleId: path }
            files = os.listdir(root)
            nodule_id = 0
            for file in files:
                nodules = os.listdir(os.path.join(root, file))
                for nodule in nodules:
                    self.data[nodule_id] = os.path.join(root, file, nodule)
                    nodule_id += 1
            nodule_ids = list(range(len(self.data)))
            train_num = int(len(self.data)*0.1)
            #test_num = len(self.data) - train_num
            np.random.shuffle(nodule_ids)
            train_ids = np.random.choice(nodule_ids, size=train_num, replace=False)
            self.train = [] # [((img1, img2), (path, None)), ((img2, img3), (None, None))]
            
            train_set = [] # [ path1, path2, path3, ... ]
            
            for i in nodule_ids:
                path = self.data[i]
                images = os.listdir(os.path.join(path, 'images'))
                #masks = os.listdir(os.path.join(path, 'mask-0'))
                images.sort(key = self.my_sort)
                #masks.sort(key = self.my_sort)
                if len(images) < 2:
                    continue
                if i in train_ids:
                    #self.train += self.data[i]
                    train_set.append(path.replace('\\', '/'))
                    mask_idx = [len(images) // 2]
                    for j in range(len(images) - 1):
                        path1 = os.path.join(path, 'images', 'slice-' + str(j) + '.png')
                        print('path1',path1)
                        path2 = os.path.join(path, 'images', 'slice-' + str(j + 1) + '.png')
                       
                        self.train += [
                                ((path1, path2)),#中间掩码
                                ((path2, path1))
                            ] 
            
            del self.data
            print('[INFO]The length of train: ', len(self.train))
            #print('[INFO]The nodule of test: ', len(self.test))
        else:
            root = self.root
            self.testdata = dict() # { noduleId: path }
            files = os.listdir(root)
            self.test = [] # [(((img1, img2, ..., img-/2), (mask1, mask2, ...)), (...))]
            test_set = [] # [ path1, path2, path3, ... ]
        
           
            for file in files:

                nodules = os.path.join(root, file)
            
              
                images = sorted(os.listdir(os.path.join(nodules, 'image')), key=lambda x: int(re.search(r'\d+', x).group()))

                masks = sorted(os.listdir(os.path.join(nodules, 'label')), key=lambda x: int(re.search(r'\d+', x).group()))
                print(os.path.join(nodules, 'label'))
                init_file=find_most_white_png(os.path.join(nodules, 'label'))
                
                # print("init_file",init_file)
                if init_file:
                    match = re.search(r'\d+', init_file)
                
                    if match:
                        init_idx = int(match.group())
                        print(init_idx)


                        # 找到 init_file 在 images 列表中的索引
                        init_idx = masks.index(init_file)

                        if len(images) < 2:
                            continue

                        init_idx = init_idx
                        backward_images = tuple([os.path.join(nodules, 'image', image) for image in reversed(images[:init_idx + 1])])
                        backward_masks = tuple([os.path.join(nodules, 'label', mask) for mask in reversed(masks[:init_idx + 1])])
                        forward_images = tuple([os.path.join(nodules, 'image', image) for image in images[init_idx:]])
                        forward_masks = tuple([os.path.join(nodules, 'label', mask) for mask in masks[init_idx:]])
                        
                        self.test.append(
                            ((backward_images, backward_masks), (forward_images, forward_masks))
                        )
    
    def process_mask(self, mask):
        mask[mask>=127.5] = 255
        mask[mask<127.5] = 0
        mask = mask
        return mask
    def pack_tensor(self, path, is_mask=False):
        path = path.replace('\\', '/').split('/')
        self.root = self.root.replace('\\','/')
        path = os.path.join(self.root, '/'.join(path[-3:]))
        
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
       

        image = torch.unsqueeze(torch.from_numpy(image), 0).type(torch.FloatTensor)

        return image

    def __getitem__(self, index):
        if self.training:
            images, mask_flags = self.train[index]
            a,b = images
            print('a',a)
            num = int(a.split('/')[9].split('-')[1].split('.')[0] )
            c,d = mask_flags
            #print(c,d)
            num_len = len(os.listdir(self.root + a.split('/')[6] + '/' +  a.split('/')[7] + '/' + a.split('/')[8]))
            imf = images[0]
            image1 = self.pack_tensor(images[0], is_mask=False)
            image2 = self.pack_tensor(images[1], is_mask=False)
            mask1 = self.pack_tensor(mask_flags[0],
                                     is_mask=True) if mask_flags[0] is not None else torch.ones_like(image1) * -1
            mask2 = self.pack_tensor(mask_flags[1],
                                     is_mask=True) if mask_flags[1] is not None else torch.ones_like(image2) * -1
            return image1, image2, mask1, mask2, num_len, num,imf
        else:
            backward, forward = self.test[index]
            backward_images = []
            backward_masks = []
            assert len(backward[0]) == len(backward[1])
            for i in range(len(backward[0])):
                backward_images.append(self.pack_tensor(backward[0][i], is_mask=False))
                backward_masks.append(self.pack_tensor(backward[1][i], is_mask=True))
            backward_images = torch.cat(backward_images, dim=0)
            backward_masks = torch.cat(backward_masks, dim=0)
            forward_images = []
            forward_masks = []
            assert len(forward[0]) == len(forward[1])
            for i in range(len(forward[0])):
                forward_images.append(self.pack_tensor(forward[0][i], is_mask=False))
                forward_masks.append(self.pack_tensor(forward[1][i], is_mask=True))

            forward_images = torch.cat(forward_images, dim=0)
            forward_masks = torch.cat(forward_masks, dim=0)
            
            print("backward_images",backward_images.shape)
            print("backward_masks",backward_masks.shape)
            print("forward_images",forward_images.shape)
            print("forward_masks",forward_masks.shape)

            assert backward_images.shape == backward_masks.shape
            assert forward_images.shape == forward_masks.shape
            
            return backward_images, backward_masks, forward_images, forward_masks


# can't split datase, must provide train_set and test_set
