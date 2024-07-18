import torch.nn.functional as F
from utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from z_order import *
import torch
import torch.nn as nn
import math

def get_relation_zorder_sample(input_u, input_v, random_sample=True, sample_size=32):
    batchsize, in_uchannels, length = input_u.shape
    _, in_vchannels, _ = input_v.shape
    device = input_u.device
    if not random_sample:
        sample_size = length

    input_u = input_u.permute(0, 2, 1)
    input_v = input_v.permute(0, 2, 1)
    ides = z_order_point_sample(input_u[:, :, :3], sample_size)

    batch_indices = torch.arange(batchsize, dtype=torch.long).to(device).view(batchsize, 1).repeat(1, sample_size)

    temp_relationu = input_u[batch_indices, ides, :].permute(0, 2, 1)
    temp_relationv = input_v[batch_indices, ides, :].permute(0, 2, 1)
    input_u = input_u.permute(0, 2, 1)
    input_v = input_v.permute(0, 2, 1)
    relation_u = torch.cat([input_u.view(batchsize, -1, length, 1).repeat(1, 1, 1, sample_size),
                            temp_relationu.view(batchsize, -1, 1, sample_size).repeat(1, 1, length, 1)], dim=1)
    relation_v = torch.cat([input_v.view(batchsize, -1, length, 1).repeat(1, 1, 1, sample_size),
                            temp_relationv.view(batchsize, -1, 1, sample_size).repeat(1, 1, length, 1)], dim=1)

    return relation_u, relation_v, temp_relationu, temp_relationv



class PointFEF(nn.Module):
    def __init__(self, in_uchannels, in_vchannels, random_sample=True, sample_size=64):
        super(PointFEF, self).__init__()

        self.random_sample = random_sample
        self.sample_size = sample_size

        self.conv_gu = nn.Conv2d(2 * in_uchannels, 2 * in_uchannels, 1)
        self.bn1 = nn.BatchNorm2d(2 * in_uchannels)

        self.conv_gv = nn.Conv2d(2 * in_vchannels, 2 * in_vchannels, 1)
        self.bn2 = nn.BatchNorm2d(2 * in_vchannels)

        self.conv_uv = nn.Conv2d(2 * in_uchannels + 2 * in_vchannels, in_vchannels, 1)
        self.bn3 = nn.BatchNorm2d(in_vchannels)

        self.conv_f = nn.Conv1d(in_vchannels, in_vchannels, 1)
        self.bn4 = nn.BatchNorm1d(in_vchannels)

    def forward(self, input_u, input_v):
        """
              Input:
                  input_u: input points position data, [B, C, N]
                  input_v: input points data, [B, D, N]
              Return:
                  new_xyz: sampled points position data, [B, C, S]
                  new_points_concat: sample points feature data, [B, D', S]
        """

        relation_u, relation_v, _, _ = get_relation_zorder_sample(input_u, input_v, random_sample=self.random_sample,
                                                                  sample_size=self.sample_size)
        relation_uv = torch.cat(
            [F.relu(self.bn1(self.conv_gu(relation_u))), F.relu(self.bn2(self.conv_gv(relation_v)))], dim=1)

        relation_uv = F.relu(self.bn3(self.conv_uv(relation_uv)))
        relation_uv = torch.max(relation_uv, 3)[0]
        relation_uv = F.relu(self.bn4(self.conv_f(relation_uv)))
        relation_uv = torch.cat([input_v + relation_uv, input_u], dim=1)

        return relation_uv


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1] 
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx


class AGConv(nn.Module):
    def __init__(self, in_channels, out_channels, feat_channels):
        super(AGConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels

        self.conv0 = nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(out_channels, out_channels * in_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y):
        
        batch_size, n_dims, num_points, k = x.size()
        y = self.conv0(y) 
        y = self.leaky_relu(self.bn0(y))
        y = self.conv1(y)  
        y = y.permute(0, 2, 3, 1).view(batch_size, num_points, k, self.out_channels,
                                       self.in_channels)  

        x = x.permute(0, 2, 3, 1).unsqueeze(4)  
        x = torch.matmul(y, x).squeeze(4)  
        x = x.permute(0, 3, 1, 2).contiguous()  

        x = self.bn1(x)
        x = self.leaky_relu(x)
        return x


class Net(nn.Module):
    def __init__(self, output_channels=3):
        super(Net, self).__init__()
        self.k = 20
        self.bn1 = nn.BatchNorm1d(3)
        self.conv1 = nn.Sequential(nn.Conv1d(64, 3, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(3 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.4)
        self.linear3 = nn.Linear(256, 3)

        self.adapt_conv1 = AGConv(3*2, 64, 3*2)

    def forward(self, x):
        batch_size = x.size(0)
        points = x 
        x, idx = get_graph_feature(x, k=self.k)

        p, _ = get_graph_feature(points, k=self.k, idx=idx)

        x = self.adapt_conv1(p, x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = self.conv1(x1)

        return x



class CAA_Module(nn.Module):
    """ Channel-wise Affinity Attention module"""

    def __init__(self, in_dim):
        super(CAA_Module, self).__init__()

        self.bn1 = nn.BatchNorm1d(256 // 4)
        self.bn2 = nn.BatchNorm1d(256 // 4)
        self.bn3 = nn.BatchNorm1d(in_dim)

        self.query_conv = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=256 // 4, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=256 // 4, kernel_size=1, bias=False),
                                      self.bn2,
                                      nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X N )
            returns :
                out : output feature maps( B X C X N )
        """
        x_hat = x.permute(0, 2, 1)
        proj_query = self.query_conv(x_hat)
        proj_key = self.key_conv(x_hat).permute(0, 2, 1)
        similarity_mat = torch.bmm(proj_key, proj_query)
        affinity_mat = torch.max(similarity_mat, -1, keepdim=True)[0].expand_as(similarity_mat) - similarity_mat
        affinity_mat = self.softmax(affinity_mat)
        proj_value = self.value_conv(x)
        out = torch.bmm(affinity_mat, proj_value)
        out = self.alpha * out + x
        return out


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(256, [0.1, 0.4], [16, 128], in_channel,
                                             [[32, 32, 64], [64, 96, 128]])

        self.PointFEF1 = PointFEF(3, 64 + 128, random_sample=True)
        self.sa2 = PointNetSetAbstraction(None, None, None, 128 + 64 + 3 + 3, [256, 512, 1024], True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

        self.bn7 = nn.BatchNorm2d(3)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 3, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.net = Net()
        self.caa1 = CAA_Module(195)
        self.caa2 = CAA_Module(3)
    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        xyz = self.net(xyz)
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l1_points = self.PointFEF1(l1_xyz, l1_points)
        l1_points = self.caa1(l1_points)

        l3_xyz, l3_points = self.sa2(l1_xyz, l1_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))

        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


if __name__ =="__main__":
    input = torch.randn(12, 6, 1024)
    model = get_model(num_class=4)
    output = model(input)
    # print(output)
    # print(model)