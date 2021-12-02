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
        padding1, padding2 = (1,1), (2,2)
        convbnrelu = lambda args : [
            nn.Conv2d(in_channels=args[0], out_channels=args[1], kernel_size=args[2], stride=args[3], padding=args[4]),
            nn.BatchNorm2d(num_features=args[1]),
            nn.ReLU()]
        self.model = nn.Sequential(
            *(convbnrelu([features3, features3, kernel3, stride1, padding1]) * 2),
            *(convbnrelu([features3, features16, kernel5, stride2, padding2])),
            *(convbnrelu([features16, features16, kernel3, stride1, padding1]) * 2),
            *(convbnrelu([features16, features32, kernel5, stride2, padding2])),
            *(convbnrelu([features32, features32, kernel3, stride1, padding1]) * 2),
            nn.Conv2d(in_channels=features32, out_channels=features32, kernel_size=kernel3, stride=stride1, padding=padding1)
        )
        print('Feature Extraction Network: ', self.model)

    def forward(self, x):
        # x: [B,3,H,W]
        # TODO
        res = self.model(x.float())
        return res

class SimilarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimilarityRegNet, self).__init__()
        # TODO
        features1, features8, features16, features32 = 1, 8, 16, 32
        stride1, stride2 = (1,1), (2,2)
        kernel3 = (3,3)
        padding_none, padding1 = 0, (1,1)
        convrelu = lambda args : [
            nn.Conv2d(in_channels=args[0], out_channels=args[1], kernel_size=args[2], stride=args[3], padding=args[4]),
            nn.ReLU()]
        self.C0 = nn.Sequential(*(convrelu([G, features8, kernel3, stride1, padding1])))
        self.C1 = nn.Sequential(*(convrelu([features8, features16, kernel3, stride2, padding1])))
        self.C3 = nn.Sequential(
            *(convrelu([features16, features32, kernel3, stride2, padding1])), # C2
            nn.ConvTranspose2d(in_channels=features32, out_channels=features16, kernel_size=kernel3, stride=stride2, padding=padding1, output_padding=padding1) # C3
        )
        self.C4 = nn.ConvTranspose2d(in_channels=features16, out_channels=features8, kernel_size=kernel3, stride=stride2, padding=padding1, output_padding=padding1)
        self.S_bar = nn.Conv2d(in_channels=features8, out_channels=features1, kernel_size=kernel3, stride=stride1, padding=padding1)
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
    HW = H*W
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        dim = rot.size()[-1]
        # meshgrid is not symmetric
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        x, y = x.contiguous(), y.contiguous()
        x, y = x.view(HW), y.view(HW)
        # TODO
        x_depth = torch.matmul(depth_values.view(B, D, 1), x.view(1, HW)).view(B, D, 1, HW)
        y_depth = torch.matmul(depth_values.view(B, D, 1), y.view(1, HW)).view(B, D, 1, HW)
        z_depth = depth_values.view(B, D, 1, 1).repeat(1, 1, 1, HW)
        homo2D = torch.cat((x_depth, y_depth, z_depth), dim=2)
        rotated = torch.matmul(rot.view(B, 1, dim, dim), homo2D)
        projected = rotated + trans.view(B, 1, dim, 1)
        index_grid = projected[:, :, :-1, :] / projected[:, :, -1:, :]
        index_grid[:, :, 0, :] = index_grid[:, :, 0, :] / ((W - 1) / 2) - 1
        index_grid[:, :, 1, :] = index_grid[:, :, 1, :] / ((H - 1) / 2) - 1
        index_grid = torch.transpose(index_grid, 2, 3).reshape(B, D * H, W, 2)

    warped_src_fea = F.grid_sample(src_fea, index_grid, mode="bilinear", align_corners=True)

    return warped_src_fea.view(B, C, D, H, W)

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO
    B, C, D, H, W = warped_src_fea.size()
    return (warped_src_fea.view(B, G, C // G, D, H, W) * ref_fea.view(B, G, C // G, 1, H ,W)).mean(2)


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
    mask = 0 < mask
    return F.l1_loss(depth_est[mask], depth_gt[mask])
