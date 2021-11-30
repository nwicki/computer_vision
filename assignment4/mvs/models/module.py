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
        self.model = nn.Sequential(
            *(convbnrelu([features3, features3, kernel3, stride1, padding_same]) * 2),
            *(convbnrelu([features3, features16, kernel5, stride2, padding2])),
            *(convbnrelu([features16, features16, kernel3, stride1, padding_same]) * 2),
            *(convbnrelu([features16, features32, kernel5, stride2, padding2])),
            *(convbnrelu([features32, features32, kernel3, stride1, padding_same]) * 2),
            nn.Conv2d(in_channels=features32, out_channels=features32, kernel_size=kernel3, stride=stride1, padding=padding_same)
        )
        print('Feature Extraction Network: ', self.model)

    def forward(self, x):
        # x: [B,3,H,W]
        # TODO
        return self.model(x.float())

class SimilarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimilarityRegNet, self).__init__()
        # TODO
        features1, features8, features16, features32 = 1, 8, 16, 32
        stride1, stride2 = (1,1), (2,2)
        kernel3 = (3,3)
        padding_same, padding_none, padding1 = 'same', 0, (1,1)
        convrelu = lambda args : [
            nn.Conv2d(in_channels=args[0], out_channels=args[1], kernel_size=args[2], stride=args[3], padding=args[4]),
            nn.ReLU()]
        self.C0 = nn.Sequential(*(convrelu([G, features8, kernel3, stride1, padding_same])))
        self.C1 = nn.Sequential(*(convrelu([features8, features16, kernel3, stride2, padding1])))
        self.C3 = nn.Sequential(
            *(convrelu([features16, features32, kernel3, stride2, padding1])), # C2
            nn.ConvTranspose2d(in_channels=features32, out_channels=features16, kernel_size=kernel3, stride=stride2, padding=padding1, output_padding=padding1) # C3
        )
        self.C4 = nn.ConvTranspose2d(in_channels=features16, out_channels=features8, kernel_size=kernel3, stride=stride2, padding=padding1, output_padding=padding1)
        self.S_bar = nn.Conv2d(in_channels=features8, out_channels=features1, kernel_size=kernel3, stride=stride1, padding=padding_same)
        print(f'Similarity RegNet: \n C0: {self.C0} \n C1: {self.C1} \n C2 & C3: {self.C3} \n C4: {self.C4} \n S_bar: {self.S_bar}' )

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO
        B, G, D, H, W = x.size()
        C0 = self.C0(torch.transpose(x, 1, 2).reshape(B * D, G, H, W))
        C1 = self.C1(C0)
        return self.S_bar(self.C4(self.C3(C1) + C1) + C0).view(B, D, H, W)


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
        xyz = torch.stack((x, y, torch.ones_like(x)))
        xyz_un = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)
        rot_xyz = torch.matmul(rot, xyz_un)

        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, D, 1) * depth_values.repeat(H * W, 1, 1, 1).view(
            B, 1, D, H * W
        )
        proj_xyz = rot_depth_xyz + trans.view(B, 3, 1, 1)
        negative_depth_mask = proj_xyz[:, 2:] <= 1e-3
        proj_xyz[:, 0:1][negative_depth_mask] = float(W)
        proj_xyz[:, 1:2][negative_depth_mask] = float(H)
        proj_xyz[:, 2:3][negative_depth_mask] = 1.0
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((W - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((H - 1) / 2) - 1
        image = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)

    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO
    warped_src_fea = F.grid_sample(src_fea, image.view(B, D * H, W, 2), 'bilinear', 'zeros', True)
    warped_src_fea = warped_src_fea.view(B, C, D, H, W)

    return warped_src_fea

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO
    B, C, D, H, W = warped_src_fea.size()
    correlation = 1 / (C // G) * (warped_src_fea.view(B, G, C // G, D, H, W) * ref_fea.view(B, G, C // G, 1, H ,W)).mean(2)
    return correlation


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    B, D, H, W = p.size()
    return torch.sum(p * depth_values.view(B, D, 1, 1), dim=1)

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    return F.l1_loss(depth_est[0 < mask], depth_gt[0 < mask])
