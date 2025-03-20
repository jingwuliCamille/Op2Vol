import torch
import numpy as np
import os
import cv2
import itertools
from model_seg_flow import Net1
from torch.utils.data import DataLoader
from load_data1 import LIDC_IDRI_flow,Chaos
from tqdm import tqdm
from model_seg_flow import net_utils
import torch, gc


pid = os.getpid()
print("当前进程的PID:", pid)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')#`设置使用的设备为GPU（如果有）或CPU。
net = Net1.OpticalSegFLow(device=device, num_channels=1, num_levels=6, use_cost_volume=True)#创建一个基于光流法和U-Net的肺部CT图像分割模型，其中OpticalSegFLow是一个自定义的类，实现了这个模型。
net.to(device)
prediction = []
output_path = '/home/zhangwei/ljw/OpticalandAffinity/result/opticalNet/Chaos'

model_checkpoint =  '/home/zhangwei/ljw/OpticalandAffinity/checkpoint6/BestModel_11_0.5116727766251365.pth'
root_data = '/raid/Data/lijingwu/Chaos'
test_dataset = Chaos(root=root_data, load=True, training=False)#加载用于测试的数据集，同样使用LIDC_IDRI_unet类。
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)#创建一个用于测试的数据加载器。
optimizer = torch.optim.Adam(list(net._pyramid.parameters()) + list(net._flow_model.parameters()), lr=1e-4)#创建一个Adam优化器，用于更新模型参数，其中_pyramid和_flow_model分别是模型中用于金字塔分层和光流计算的两个子模块。



net.load_state_dict(torch.load(model_checkpoint))

def test():
#这是定义一个名为 "test" 的函数。
    with torch.no_grad():
    #使用 "with" 语句打开一个上下文，在这个上下文中进行的计算不会被计算图所跟踪。这意味着在这个上下文中的变量的梯度不会被计算和更新，这将减少内存占用并提高代码运行速度。
        net.eval()
    #将神经网络设置为测试模式，这将关闭训练中使用的一些技术，如 dropout 和 batch normalization 等。
        photo_loss = []
        dice = []
    #定义两个空列表，分别用于存储损失和 Dice 分数的值
        bar = tqdm(test_loader)
    #使用 tqdm 包装测试数据加载器，以便在命令行中以进度条的形式显示数据加载进度。
        for step, (backward_images, backward_masks, forward_images, forward_masks) in enumerate(bar):
    #使用 for 循环逐步加载测试数据，其中每个数据都由 backward_images、backward_masks、forward_images 和 forward_masks 四个部分组成。step 是一个计数器，用于跟踪测试数据的数量。
            bar.set_description('Test')
            #设置进度条的文本描述为 "Test"。
            # backward and forward
            images_warp = [[], []]
            masks_warp = [[], []]
            flows_vis = [[], []]
            #定义三个空列表，分别用于存储图像、掩码和光流的结果。
            images = [backward_images, forward_images]
            print("backward_images.shape",backward_images.shape)
            print("forward_images.shape",forward_images.shape)
            masks = [backward_masks, forward_masks]
        #将 backward_images 和 forward_images 分别存储在名为 images 的列表中，将 backward_masks 和 forward_masks 分别存储在名为 masks 的列表中。
            for idx in range(len(images)):
                B, slices, H, W = images[idx].shape
                if slices < 2:
                    continue
            #遍历 images 列表，并检查每个元素中的 "slices" 维度是否小于 2。如果是，则跳过当前循环。
                img1 = images[idx][:, 0:1, :, :].to(device)
                
                mask1 = masks[idx][:, 0:1, :, :].to(device)
                images_warp[idx].append(img1)
                
                masks_warp[idx].append(mask1)
            #从输入数据中提取第一张图像和掩码，并将它们移动到指定的设备上。
            #将图像和掩码添加到指定索引的列表中，以便后续的可视化和评估。
                for i in range(slices - 1):
                    #遍历每个时间步，并进行光流前向计算。
                    img1 = images[idx][:, i:(i+1), :, :].to(device)
                    
                    img2 = images[idx][:, (i+1):(i+2), :, :].to(device)
                #从输入数据中提取当前时间步和下一个时间步的图像，并将它们移动到指定的设备上。
                    mask2 = masks[idx][:, (i+1):(i+2), :, :].to(device)
                #从输入数据中提取当前时间步的掩码，并将其移动到指定的设备上。
                    flows_f, segs_f, fp2, fp1, flows_b, segs_b,occlusion = net(img2, img1 , training=False)
                #使用输入图像计算前向和后向光流，以及前向和后向分割分支的输出。此外，这里的net指代的是神经网络模型。
                    warp = net_utils.flow_to_warp(flows_f[0])
                    #将光流转换为位移场。
                    warped_img1 = net_utils.resample(img1, warp)
                    warped_mask1 = net_utils.resample(mask1, warp)
                #使用位移场对图像和掩码进行重采样，以获取在当前时间步的图像中的对应像素位置。
                    loss = net_utils.compute_l1_loss(img2, warped_img1, flows_f, use_mag_loss=False)                   
                #计算前向光流的L1损失。
                    photo_loss.append(loss.item())
                #将当前图像的L1损失添加到列表中，以便后续的平均值计算。

                    is_all_zero = torch.all(mask2 == 0).item()
                    if not is_all_zero:
                        dice.append(net_utils.mask_dice(
                                net_utils.norm_mask(warped_mask1.clone()),
                                mask2
                            ))
                    else:
                        dice.append(net_utils.mask_dice(
                                mask2,
                                mask2
                            ))
                   
                    # dice.append(net_utils.mask_dice(
                    #             net_utils.norm_mask(warped_mask1.clone()),
                    #             mask2))
                    

                #计算分割结果的Dice系数，并将其添加到指定列表中，以便后续的平均值计算。
                    images_warp[idx].append(warped_img1)
                    masks_warp[idx].append(warped_mask1)
                    flows_vis[idx].append(flows_f[0])
                #将重采样后的图像和掩码添加到指定索引的列表中，以便后续的可视化和评估。
                #将前向光流添加到指定索引的列表中，以便后续的可视化。
                    mask1 = warped_mask1
            # ori_iamges
            #images_warp = [[backward_images[:, i:i+1, :, :] for i in range(backward_images.shape[1])],
            #               [forward_images[:, i:i+1, :, :] for i in range(forward_images.shape[1])]]
            
            save_test_result(step, images_warp, masks_warp, flows_vis)
            print("photo_loss",np.mean(photo_loss),"dice:",np.mean(dice))
        return np.mean(photo_loss), np.mean(dice)

def save_test_result(step, images_warp, masks_warp, flows_vis):
    
    path =root_data
    pathlist = os.listdir(path)    
    print(len(pathlist))
    mid = len(images_warp[0])
    end = len(images_warp[0])+len(images_warp[1])

    #print(path)
    #print('ll;',len(slices))
    #print(output_path)
    if not os.path.exists(os.path.join(output_path)):
        os.mkdir(os.path.join(output_path))
    if not os.path.exists(os.path.join(output_path, pathlist[step])):
        os.mkdir(os.path.join(output_path, pathlist[step]))
    path = os.path.join(output_path, pathlist[step])
    if not os.path.exists(os.path.join(path, 'images')):
        os.mkdir(os.path.join(path, 'images'))
    if not os.path.exists(os.path.join(path, 'mask-0')):
        os.mkdir(os.path.join(path, 'mask-0'))
    if not os.path.exists(os.path.join(path, 'flows')):
        os.mkdir(os.path.join(path, 'flows'))
    for idx in range(len(images_warp)):
        #print('lllllll',len(images_warp))
        #print('idxi',idx)
        if idx == 0:
            # backward
            appendix = list(range(mid, -1, -1))
            #print('len1',len(appendix))
        else:
            # forward
            appendix = list(range(mid, end, 1))
            
        print("imglen",len(images_warp[idx]))
        print("masklen",len(masks_warp[idx]))
        print("appendixlen",len(appendix))
       
        for i in range(len(images_warp[idx])):
            ap = appendix[i]
            image = torch.squeeze(images_warp[idx][i]).cpu().numpy()
            filename = 'slice-' + str(ap) + '.png' if ap != mid else 'slice-' + str(ap) + '-ori.png'
            cv2.imwrite(os.path.join(path, 'images', filename), image)
            mask = net_utils.denorm_mask(torch.squeeze(masks_warp[idx][i]).cpu().numpy())
            filename = 'slice-' + str(ap) + '.png' if ap != mid else 'slice-' + str(ap) + '-ori.png'
            cv2.imwrite(os.path.join(path, 'mask-0', filename), mask)
            #cv2.imread(os.path.join(path, 'mask-0', filename), mask)
            if i < len(flows_vis[idx]):
                flow = torch.squeeze(flows_vis[idx][i]).cpu().numpy()
                flow_color = flow_vis_fn(np.transpose(flow, axes=(1, 2, 0)))
                # filename = 'flow-' + str(appendix[i]) + '-' + str(appendix[i + 1]) + '.png'
                cv2.imwrite(os.path.join(path, 'flows', filename), flow_color)

def flow_vis_fn(flow):
    hsv = np.zeros((128, 128, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_color
                #cv2.imwrite(os.path.join(path, 'flows', filename), flow_color)
#函数参数包括step、images_warp、masks_warp和flows_vis，这些参数都是在测试过程中生成的。
# 其中，step是当前测试步骤的编号，images_warp和masks_warp是图像和掩模的变形结果，flows_vis是可视化的光流结果。
#在函数内部，首先根据当前测试的路径解析出nodule_id和LIDC_ID，并根据这些信息创建了对应的目录，用于存储测试结果。
#接下来，对于每个变形后的图像，将其保存为PNG格式的图像文件，并将其存储在images目录中。
# 对于每个变形后的掩模，也将其保存为PNG格式的图像文件，并将其存储在mask-0目录中。对于每个可视化的光流，将其保存为PNG格式的图像文件，并将其存储在flows目录中。
#最后返回的是平均的photo_loss和dice。     

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    torch.cuda.set_device(2)

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()#清除PyTorch缓存。
    photo_loss,dice = test()
    print('photo_loss ' , photo_loss ,'dice',dice)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # net = Net1.OpticalSegFLow(device=device, num_channels=1, num_levels=6, use_cost_volume=True)#创建一个基于光流法和U-Net的肺部CT图像分割模型，其中OpticalSegFLow是一个自定义的类，实现了这个模型。
    # net.to(device)
    
    # img1 = torch.randn(1,1,128, 128).cuda()
    # img2 = torch.randn(1,1,128, 128).cuda()
    # img1 = img1.to(device)
    # img2 = img2.to(device)
    
    # flows_f, segs_f, fp2, fp1, flows_b, segs_b,occlusion = net(img2, img1 , training=False)
    # print(flows_f)