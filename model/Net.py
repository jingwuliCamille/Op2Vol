from model_seg_flow import FeaturePath
from model import FlowPath
from model import net_utils
import torch.nn as nn

class OpticalFLow(nn.Module):
    def __init__(self, device, num_channels=1, num_levels = 5, use_cost_volume=True):
        super(OpticalFLow, self).__init__()
        
        self.device = device

        self._pyramid = FeaturePath.FeaturePath(num_levels=num_levels, num_channels=num_channels)
        self._flow_model = FlowPath.FlowPath(device = device, num_levels = num_levels, 
                                               num_channels_upsampled_context=32,
                                               use_cost_volume=use_cost_volume, use_feature_warp=True)

    def forward(self, img1, img2, training=True):
        """Compute the flow between image pairs
        :param img1: [BCHW] image 1 batch
        :param img2: [BCHW] image 2 batch
        :return: [B2HW] flow tensor for each image pair in batch, feature pyramid list for img pairs
        """
        fp1 = self._pyramid(img1)
        fp2 = self._pyramid(img2)
        flow_forward = self._flow_model(fp1, fp2, training)
        flow_backward = self._flow_model(fp2, fp1, training)
        occlusions = net_utils.occlusion_mask(flow_forward, flow_backward, range(len(flow_forward)))
        return flow_forward, occlusions, fp1, fp2

    def compute_loss(self, img1, img2, features1, features2, occlusions, flows):
        f1 = [img1] + features1
        f2 = [img2] + features2

        warps = [net_utils.flow_to_warp(f) for f in flows]
        warped_f2 = [net_utils.resample(f, w) for (f, w) in zip(f2, warps)]
        
        #compute_level = range(len(flows))
        compute_level = [0]
        loss_feature_consistency = net_utils.loss_feature_consistency(f1, warped_f2, flows,
                                                                      occlusions, compute_level)
        loss_smoothness = net_utils.loss_smoothness(img1, flows, compute_level)
        loss_ssim_weight = net_utils.loss_ssim_weight(img1, img2, flows, compute_level,
                                                      occlusions, self.device)

        #loss = net_utils.compute_all_loss(f1, warped_f2, flows, self.device)
        loss = 10 * loss_feature_consistency + loss_smoothness + loss_ssim_weight
        return loss