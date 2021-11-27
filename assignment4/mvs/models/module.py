import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # TODO
        features3, features16, features32 = 3, 16, 32
        stride1, stride2 = (1,1), (2,2)
        kernel3, kernel5 = (3,3), (5,5)
        padding_same, padding2 = 'same', 2
        convbnrelu = lambda args : [
            nn.Conv2d(in_channels=args[0], out_channels=args[1], kernel_size=args[2], stride=args[3], padding=args[4]),
            nn.BatchNorm2d(num_features=args[1]),
            nn.ReLU()]
        test = [*(convbnrelu([features3, features3, kernel3, stride1, padding_same]) * 2)]
        print(test)
        exit()
        self.model = lambda x : x
        # self.model = nn.Sequential[
        #     *(convbnrelu([features3, features3, kernel3, stride1, padding_same]) * 2),
        #     *(convbnrelu([features3, features16, kernel5, stride2, padding2])),
        #     *(convbnrelu([features16, features16, kernel3, stride1, padding_same]) * 2),
        #     *(convbnrelu([features16, features32, kernel5, stride2, padding2])),
        #     *(convbnrelu([features32, features32, kernel3, stride1, padding_same]) * 2),
        #     nn.Conv2d(in_channels=features32, out_channels=features32, kernel_size=kernel3, stride=stride1, padding=padding_same)
        # ]
        print('Feature Extraction Network: ', self.model)

    def forward(self, x):
        # x: [B,3,H,W]
        # TODO
        return self.model(x)

class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO
        return None


def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        # TODO


    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO

    warped_src_fea = None
    return warped_src_fea

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO
    return None


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    return None

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    return None
