import torch.nn as nn
import torch
import torch.nn.functional as func


from model_seg_flow import net_utils
from model_seg_flow import FeaturePath
# import net_utils


class FlowSegPath(nn.Module):
    def __init__(self,
                 device,
                 leaky_relu_alpha=0.1,
                 dropout_rate=0.25,
                 num_channels_upsampled_context=32,
                 num_levels=6,
                 seg_num_levels=4,
                 normalize_before_cost_volume=True,
                 channel_multiplier=1.,
                 use_cost_volume=True,
                 use_feature_warp=True,
                 accumulate_flow=True,
                 init_flow = False,
                 dual_seg = True,
                 archi = [0, 1, 0, 1, 1, 1]):
        super(FlowSegPath, self).__init__()
        
        self._device = device
        self._leaky_relu_alpha = leaky_relu_alpha #0.1
        self._drop_out_rate = dropout_rate#0.25
        self._num_context_up_channels = num_channels_upsampled_context#32
        self._num_levels = num_levels#6
        self._seg_num_levels = seg_num_levels#4
        self._normalize_before_cost_volume = normalize_before_cost_volume#true
        self._channel_multiplier = channel_multiplier#1
        self._use_cost_volume = use_cost_volume#true
        self._use_feature_warp = use_feature_warp#true
        self._accumulate_flow = accumulate_flow#true
        self._init_flow = init_flow#false
        self._dual_seg = dual_seg#true
        self._archi = archi

        self._refine_model = self._build_refinement_model()
        self._flow_layers = self._build_flow_layers()
        self._seg_model = self._build_segment_model()
        if not self._use_cost_volume:
            self._cost_volume_surrogate_convs = self._build_cost_volume_surrogate_convs()
        if self._num_context_up_channels:
            self._context_up_layers = self._build_upsample_layers(
                num_channels=int(self._num_context_up_channels * channel_multiplier))#32

    def forward(self, feature_pyramid1, feature_pyramid2, training=True):
        """Run the model."""
        context = None
        flow = None
        flow_up = None #光流场
        context_up = None
        flows = []
        
        seg_context1 = None
        seg_context2 = None
        seg1 = None
        seg2 = None
        seg1_up = None
        seg2_up = None
        segs = [[],[]]

        # Go top down through the levels to the second to last one to estimate flow.
        for level, (features1, features2) in reversed(
                list(enumerate(zip(feature_pyramid1, feature_pyramid2)))[:]):

            # init flows with zeros for coarsest level if needed
            if self._init_flow and flow_up is None:
                batch_size, _, height, width = features1.shape
                flow_up = torch.zeros([batch_size, 2, height, width]).to(self._device)
                if self._num_context_up_channels:
                    num_channels = int(self._num_context_up_channels)#32
                    context_up = torch.zeros([batch_size, num_channels, height, width]).to(self._device)#[b,32,h,w]

            # Warp features2 with upsampled flow from higher level.
            if flow_up is None or not self._use_feature_warp:
                warped2 = features2
            else:# if flow_up is true
                warp_up = net_utils.flow_to_warp(flow_up)
                warped2 = net_utils.resample(features2, warp_up)

            # Compute cost volume by comparing features1 and warped features2.
            features1_normalized, warped2_normalized = net_utils.normalize_features(
                [features1, warped2],
                normalize=self._normalize_before_cost_volume,
                center=self._normalize_before_cost_volume,
                moments_across_channels=True,
                moments_across_images=True)

            if self._use_cost_volume:
                cost_volume = net_utils.compute_cost_volume(features1_normalized, warped2_normalized, max_displacement = 4)
            else:
                concat_features = torch.cat([features1_normalized, warped2_normalized], dim=1)
                cost_volume = self._cost_volume_surrogate_convs[level](concat_features)

            cost_volume = func.leaky_relu(cost_volume, negative_slope=self._leaky_relu_alpha)
            
            
            if level <= self._num_levels - self._seg_num_levels + 1:#3
                seg_in = None
                if seg1_up is None:
                    seg_in = torch.cat([context_up, flow_up, features1], dim=1)
                else:
                    seg_in = torch.cat([context_up, flow_up, seg1_up, features1], dim=1)
                seg_layers = self._seg_model[level]
                for layer in seg_layers[:-1]:
                    seg_out = layer(seg_in)
                    seg_in = torch.cat([seg_in, seg_out], dim=1)
                seg_context1 = seg_out
                
                seg1 = seg_layers[-1](seg_context1)
                if self._archi[level] == 1:
                    seg1_up = net_utils.upsample(seg1, is_flow=False)
                else:
                    seg1_up = seg1
                
                segs[0].insert(0, seg1)
                
                if self._dual_seg:
                    seg_in = None
                    if seg2_up is None:
                        seg_in = torch.cat([context_up, flow_up, features2], dim=1)
                    else:
                        seg_in = torch.cat([context_up, flow_up, seg2_up, features2], dim=1)
                    seg_layers = self._seg_model[level]
                    for layer in seg_layers[:-1]:
                        seg_out = layer(seg_in)
                        seg_in = torch.cat([seg_in, seg_out], dim=1)
                    seg_context2 = seg_out
                    
                    seg2 = seg_layers[-1](seg_context2)
                    if self._archi[level] == 1:
                        seg2_up = net_utils.upsample(seg2, is_flow=False)
                    else:
                        seg2_up = seg2
                    
                    segs[1].insert(0, seg2)

            # Compute context and flow from previous flow, cost volume, and features1.
            if flow_up is None:
                x_in = torch.cat([cost_volume, features1], dim=1)
            else:
                if context_up is None:
                    x_in = torch.cat([flow_up, cost_volume, features1], dim=1)
                else:
                    x_in = torch.cat([context_up, flow_up, cost_volume, features1], dim=1)
            
            if seg_context1 is not None and seg1 is not None:
                if self._dual_seg:
                    x_in = torch.cat([x_in, seg_context1, seg1, seg_context2, seg2], dim=1)
                else:
                    x_in = torch.cat([x_in, seg_context1, seg1], dim=1)

            # Use dense-net connections.
            x_out = None
            flow_layers = self._flow_layers[level]
            print(flow_layers)
            for layer in flow_layers[:-1]:
                x_out = layer(x_in)
                x_in = torch.cat([x_in, x_out], dim=1)
            context = x_out

            flow = flow_layers[-1](context)

            # dropout full layer
            if training and self._drop_out_rate:
                maybe_dropout = (torch.rand([]) > self._drop_out_rate).type(torch.get_default_dtype())
                # note that operation must not be inplace, otherwise autograd will fail pathetically
                context = context * maybe_dropout
                flow = flow * maybe_dropout

            if flow_up is not None and self._accumulate_flow:
                flow += flow_up

            # Upsample flow for the next lower level.
            if self._archi[level] == 1:
                flow_up = net_utils.upsample(flow, is_flow=True)
                if self._num_context_up_channels:
                    context_up = self._context_up_layers[level](context)
            else:
                flow_up = flow
                context_up = context

            # Append results to list.
            flows.insert(0, flow)

        return flows, segs

    def _build_cost_volume_surrogate_convs(self):
        layers = nn.ModuleList()
        for _ in range(self._num_levels):#6
            layers.append(nn.Sequential(
                nn.ZeroPad2d((2,1,2,1)), # should correspond to "SAME" in keras
                nn.Conv2d(
                    in_channels=int(2 * self._num_channels_upsampled_context),
                    out_channels=int(2 * self._num_channels_upsampled_context),
                    kernel_size=(4, 4)))
            )
        return layers

    def _build_upsample_layers(self, num_channels):
        """Build layers for upsampling via deconvolution."""
        layers = []
        for unused_level in range(self._num_levels):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1))
        return nn.ModuleList(layers)

    def _build_flow_layers(self):
        """Build layers for flow estimation."""
        # Empty list of layers level 0 because flow is only estimated at levels > 0.
        #result = nn.ModuleList([nn.ModuleList()])
        result = nn.ModuleList()

        block_layers = [128, 128, 96, 64, self._num_context_up_channels]#128，128，96，64，32

        for i in range(0, self._num_levels):#6
            layers = nn.ModuleList()
            last_in_channels = (64+32) if not self._use_cost_volume else (81+32)
            if i != self._num_levels-1:#6
                last_in_channels += 2 + self._num_context_up_channels
            if i <= self._num_levels - self._seg_num_levels + 1:#3
                if self._dual_seg:
                    last_in_channels += 2 * (self._num_context_up_channels + 1)
                else:
                    last_in_channels += self._num_context_up_channels + 1

            for c in block_layers:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=last_in_channels,
                            out_channels=int(c),
                            kernel_size=(3, 3),
                            padding=1),
                        nn.LeakyReLU(
                            negative_slope=self._leaky_relu_alpha)
                    ))
                last_in_channels += int(c)
            layers.append(
                nn.Conv2d(
                    in_channels=block_layers[-1],
                    out_channels=2,
                    kernel_size=(3, 3),
                    padding=1))
            result.append(layers)
        return result

    def _build_refinement_model(self):
        """Build model for flow refinement using dilated convolutions."""
        layers = []
        last_in_channels = self._num_context_up_channels + 2
        for c, d in [(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)]:
            layers.append(
                nn.Conv2d(
                    in_channels=last_in_channels,
                    out_channels=int(c),
                    kernel_size=(3, 3),
                    stride=1,
                    padding=d,
                    dilation=d))
            layers.append(
                nn.LeakyReLU(negative_slope=self._leaky_relu_alpha))
            last_in_channels = int(c)
        layers.append(
            nn.Conv2d(
                in_channels=last_in_channels,
                out_channels=2,
                kernel_size=(3, 3),
                stride=1,
                padding=1))
        return nn.ModuleList(layers)
    
    def _build_segment_model(self):
        modules = nn.ModuleList()
        
        block_layers = [128, 128, 96, 64, self._num_context_up_channels]
        
        for i in range(0, self._seg_num_levels):
            layers = nn.ModuleList()
            if i != self._seg_num_levels - 1:
                last_in_channels = 32 + self._num_context_up_channels + 2 + 1
            else:
                last_in_channels = 32 + self._num_context_up_channels + 2

            for c in block_layers:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=last_in_channels,
                            out_channels=int(c),
                            kernel_size=(3, 3),
                            padding=1),
                        nn.BatchNorm2d(int(c)),
                        nn.LeakyReLU(
                            negative_slope=self._leaky_relu_alpha)
                    ))
                last_in_channels += int(c)
            layers.append(
                nn.Conv2d(
                    in_channels=block_layers[-1],
                    out_channels=1,
                    kernel_size=(3, 3),
                    padding=1))
            modules.append(layers)
        return modules
        
        
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    FeatureModule = FeaturePath(device=device, in_channels_img=1)
    img1 = torch.rand(1, 1, 128, 128).to(device)
    img2 = torch.rand(1, 1, 128, 128).to(device)
    fp1 = FeatureModule(img1)
    fp2 = FeatureModule(img2)
    
    flow_model=FlowSegPath(device = device, num_levels = 6, seg_num_levels=4,num_channels_upsampled_context=32,
                                               use_cost_volume=True, use_feature_warp=True)
    flow_forward, seg_forward =flow_model(fp1, fp2, True)
            
        
        
        
        
        


