import torch
settings = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'weight_smooth1': 0.0001,
    'smoothness_edge_constant': 100.,
    'weight_ssim': 0.5
}