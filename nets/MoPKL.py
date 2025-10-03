import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from einops import rearrange
import matplotlib.pyplot as plt
import cv2
import cupy as cp

class MotionModel(nn.Module):
    # 初始化，超参数分别控制输入维度、潜在空间大小和隐藏层
    def __init__(self, text_input_dim=130*300, latent_dim=128, hidden_dim=1024): 
        super(MotionModel, self).__init__()
        self.latent_dim = latent_dim

        # Text Conditioner 
        self.text_fc1     = nn.Linear(text_input_dim, hidden_dim) # 定义了一个全连接层，把高维文本嵌入映射到隐藏层
        self.text_fc_mu   = nn.Linear(hidden_dim, latent_dim) # 定义了一个全连接层，把中间表示投影成潜在分布的均值向量
        self.text_fc_logvar = nn.Linear(hidden_dim, latent_dim) # 定义了一个全连接层，输出潜在分布的对数方差
        # 在上面的代码中，我们将文本描述对应一个分布，表示其中的不确定性（高斯分布），再从里面随机采样潜在向量。这样能表达描述里的模糊性与不确定性，提升泛化能力。

        # Motion Encoder
        self.visual_conv = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, padding=1) # 降维度33卷积
        self.visual_bn   = nn.BatchNorm2d(32) # 对每个通道做批归一化
        self.visual_fc   = nn.Linear(32 * 64 * 64, latent_dim) # 全连接层，生成视觉特征潜在向量，与文本模态对齐

        # Decoder
        self.heatmap_fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, 
                                          kernel_size=4, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, 
                                          kernel_size=4, stride=4, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, 
                                          kernel_size=4, stride=4, padding=0)
        self.bn3 = nn.BatchNorm2d(1) # 把潜在向量投影回二维空间（反卷积）

    @staticmethod
    # 生成潜在向量
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 根据潜在向量得到运动热力图，表示目标在每个像素位置的可能性
    def decode_heatmap(self, z):
        B = z.size(0)
        x = F.relu(self.heatmap_fc(z))       
        x = x.view(B, 256, 8, 8)             

        x = self.deconv1(x)                
        x = self.bn1(x)
        x = F.relu(x)

        x = self.deconv2(x)                
        x = self.bn2(x)
        x = F.relu(x)

        x = self.deconv3(x)                 
        x = self.bn3(x)
        heatmap = torch.sigmoid(x)          
        heatmap = heatmap.squeeze(1)         
        return heatmap

    def train_forward(self, motion_prior, feats, descriptions, alpha=1.0, beta=1.0):
        B = descriptions.size(0)
        
        # --- Text Branch --- 利用变分自编码器（VAE）来表达运动描述的不确定性，并解码成二维运动热力图
        x_text = descriptions.view(B, -1) # 展平二维文本描述
        h_text = F.relu(self.text_fc1(x_text)) # 把文本描述映射到隐藏层
        mu = self.text_fc_mu(h_text) # 把隐藏层映射为潜在分布的均值向量
        logvar = self.text_fc_logvar(h_text) # 映射为潜在分布的对数方差
        z = self.reparameterize(mu, logvar) # 把文本描述映射为潜在向量
        kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)
        kl_loss = kl_loss / (B * mu.size(1)) # KL散度损失，使后验分布接近标准正态分布，防止过拟合
        motion_text = self.decode_heatmap(z) # 生成运动热力图
        text_recon_loss = F.mse_loss(motion_text, motion_prior) # 计算运动描述与热力图的均方误差，使解码热力图尽可能与运动描述一致

        # --- Visual Branch ---
        x_vis = self.visual_conv(feats)   
        x_vis = self.visual_bn(x_vis)
        x_vis = F.relu(x_vis)
        x_vis = x_vis.view(B, -1)          
        z_prime = self.visual_fc(x_vis) # 视觉特征图 卷积、BN、激活、展平、全连接，得到视觉潜在向量
        motion_vis = self.decode_heatmap(z_prime)
        visual_recon_loss = F.mse_loss(motion_vis, motion_prior) # 计算运动描述与视觉热力图的均方误差
        
        # Text Branch是语言描述到潜在向量再到热力图的重建过程，其中运用到了VAE来丰富语言描述的不确定性，使模型能学习到“潜在运动模式”的分布
        # Visual Branch是视觉特征到潜在向量再到热力图的重建过程，它直接由骨干网络提取的视觉特征得到 
        
        latent_consistency_loss = F.mse_loss(z_prime, z) # 对齐文本潜在向量与视觉潜在向量

        # --- Alignment Loss ---
        loss = text_recon_loss + visual_recon_loss + alpha * latent_consistency_loss + beta * kl_loss
        # 总损失包括：文本重建损失、视觉重建损失、潜在向量对齐损失、KL散度损失
        return motion_vis, loss
        
    # 推理过程，直接由视觉特征得到热力图
    def inference_forward(self, feats):
        B = feats.size(0)
        x_vis = self.visual_conv(feats) 
        x_vis = self.visual_bn(x_vis)
        x_vis = F.relu(x_vis)
        x_vis = x_vis.view(B, -1)
        z_prime = self.visual_fc(x_vis)    
        motion = self.decode_heatmap(z_prime)
        return motion


class GraphAttentionLayer(nn.Module):
    # 单头GAT
    def __init__(self, in_dim, out_dim, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False) # 把节点输入维度映射到输出维度
        self.a = nn.Parameter(torch.zeros(size=(2*out_dim, 1))) # 注意力参数向量
        nn.init.xavier_uniform_(self.a.data, gain=1.414) # xacier初始化
        self.leaky_relu = nn.LeakyReLU(alpha)
        
    def forward(self, h, adj):
        B, N, _ = h.size()
        Wh = self.W(h) # 对所有节点特征做线性变换
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1) 
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1) # 复制节点ij的特征
        Wh_cat = torch.cat([Wh_i, Wh_j], dim=-1) # 拼接特征，得到(B, N, N, 2*out_dim)，表示每一对节点ij的特征对
        e = torch.matmul(Wh_cat, self.a).squeeze(-1)    
        e = self.leaky_relu(e) # 用参数向量a对特征对做投影，得到注意力分数
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) # 只保留邻接矩阵里存在的边的分数，确保softmax时只在相连节点归一化
        attention = F.softmax(attention, dim=-1) # 对每个节点i，归一化所有邻居j的注意力系数
        h_prime = torch.matmul(attention, Wh) # 聚合邻居特征，更新节点表示
        
        return F.elu(h_prime)


class MultiHeadGATLayer(nn.Module):
    # 多头GAT
    def __init__(self, in_dim, out_dim, num_heads=4, alpha=0.2): # 4个注意力头，每个都有独立的权重矩阵和注意力向量
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_dim, out_dim, alpha=alpha) 
            for _ in range(num_heads)
        ])
        
    def forward(self, h, adj):
        out = [head(h, adj) for head in self.heads] # 对每个注意力头执行一次前向传播
        out = torch.cat(out, dim=-1) # 多头注意力特征拼接
        return out

class GATNet(nn.Module):
    # 叠加两个多头注意力层
    def __init__(self, in_dim=1024, hidden_dim=128, out_dim=256, 
                 num_heads_1=8, num_heads_2=8, alpha=0.2):
        super(GATNet, self).__init__()
        self.gat1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads=num_heads_1, alpha=alpha)
        self.gat2 = MultiHeadGATLayer(hidden_dim * num_heads_1, out_dim, num_heads=num_heads_2, alpha=alpha)
        
    def forward(self, x, adj):
        x = self.gat1(x, adj)  
        x = self.gat2(x, adj)  
        return x

 

class Feature_Extractor(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act) # CSPDarknet主干，输出dark3/dark4/dark5分别表示：中分辨率/低分辨率/更低分辨率
        self.in_features    = in_features
        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest") # 上采样最高层特征
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act) # 1*1卷积，通道1024压缩到512
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        ) # 把上采样后的高层特征（dark5）与dark4做CSP融合
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        ) # 把上采样后的dark45融合特征与dark3进一步融合

    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]
        P5          = self.lateral_conv0(feat3)
        P5_upsample = self.upsample(P5)
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        P5_upsample = self.C3_p4(P5_upsample)
        P4          = self.reduce_conv1(P5_upsample) 
        P4_upsample = self.upsample(P4) 
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        P3_out      = self.C3_p3(P4_upsample)  
        
        return P3_out


class MoPKL(nn.Module):
    def __init__(self, num_classes, num_frame=5):
        super(MoPKL, self).__init__()
        
        self.num_frame = num_frame # 两帧输入
        self.backbone = Feature_Extractor(0.33,0.50)
        self.fusion = Fusion_Module(channels=[128], num_frame=num_frame)
        self.head = YOLOXHead(num_classes=num_classes, width = 1.0, in_channels = [256], act = "silu")
        self.conv_vl = nn.Sequential(
            BaseConv(128*2,256,3,1),
            BaseConv(256,256,3,1),
            BaseConv(256,256,1,1))
        self.conv_m = nn.Sequential(
            BaseConv(1,64,3,2),
            BaseConv(64,128,3,2),
            BaseConv(128,256,3,2),
            BaseConv(256,256,1,1))
        self.vf = nn.Sequential(
            BaseConv(256,16,3,2),
            BaseConv(16,16,1,1))

        """
        text_input_dim
        
        ITSDT: 130*300
        DAUB-R: 20*300
        IRDST-H: 20*300
        """
        self.motion = MotionModel(text_input_dim=130*300, latent_dim=128, hidden_dim=1024)

        
        self.GAT = GATNet(
                            in_dim=1024, 
                            hidden_dim=128, 
                            out_dim=512, 
                            num_heads_1=2, 
                            num_heads_2=2, 
                            alpha=0.2
                        )
        self.m1 = nn.Sequential(
            BaseConv(16,64,3,1),
            BaseConv(64,128,3,1),
            BaseConv(128,128,1,1))
        self.m2 = nn.Linear(1024,4096)
        
    def forward(self, inputs, descriptions=None, multi_targets=None, relation=None):
        feat = []
        outputs = []
        
        for i in range(self.num_frame-2,self.num_frame): # 输入序列两帧
            f_feats = self.backbone(inputs[:,:,i,:,:]) # 提取特征，inputs参数分别为：Batch size, Channel, 第几帧, Height, Width
            feat.append(f_feats) # 把特征存到列表feat
        B, N, W, H = f_feats.shape
        feats = self.conv_vl(torch.cat(feat,1)).squeeze(1) # 把相邻两帧在通道拼接后做卷积融合

        if self.training: 
            # Language-Driven Motion Alignment
            multi_targets = [mt.cuda() if isinstance(mt, torch.Tensor) else mt for mt in multi_targets]
            motion_prior = generate_motion(multi_targets, inputs,
                                                base_kernel_length=1, length_scale=3,
                                                kernel_width=21, alpha=0.1, sigma=3,
                                                brightness_factor=0.1, motion_threshold=1e-2) # 得到运动先验热力图
            motion_prior_cpu = [cp.asnumpy(h) if hasattr(h, 'ndim') else h for h in motion_prior]
            motion_prior = torch.tensor(np.stack(motion_prior_cpu)).cuda() # 转成torch sensor
            motion, loss_alignment = self.motion.train_forward(motion_prior, feats, descriptions[:,-1,:,:], alpha=1.0, beta=1.0) 
            # 输入参数分别代表：运动先验热力图（由实际框得到），最后两帧融合特征，最后一帧的语言描述

            # Motion-Relation Learning
            v_feat = self.vf(feats) # 视觉特征转换
            v_feat = rearrange(v_feat, 'b n w h -> b n (w h)', b=B, n=16, w=32, h=32) # 转换成节点表示，把每个空间位置当作一个图节点，每个节点n维特征
            h = self.GAT(v_feat, relation.squeeze(1)) # 用节点特征和邻接矩阵建图
            h_i = h.unsqueeze(2) 
            h_j = h.unsqueeze(1)
            pred_relation = torch.sum(h_i * h_j, dim=-1) # 通过内积计算两个节点间相似度
            loss_relation = F.mse_loss(pred_relation, relation.squeeze(1))
        else:
            motion = self.motion.inference_forward(feats)

        motion = self.conv_m(motion.unsqueeze(1)) # 将视觉运动热力图转换成张量
        feat = self.fusion(motion, feat[-1]) # 
        outputs  = self.head(feat) 
        
        if self.training:
            return outputs, loss_alignment + loss_relation
        else:
            return outputs


            

def add_comet_kernel(heatmap, head_center, tail_angle, kernel_length=61, kernel_width=21, 
                     alpha=0.1, sigma=3, brightness_scale=1.0):
    # 在目标位置，沿运动方向，画出一条亮度逐渐衰减、横向高斯扩散的“彗星尾巴”
    half_width = (kernel_width - 1) / 2.0
    u = cp.arange(kernel_length).reshape(kernel_length, 1)  
    v = cp.arange(kernel_width).reshape(1, kernel_width) - half_width 
    weight = brightness_scale * cp.exp(-alpha * u) * cp.exp(-(v**2) / (2 * sigma**2))
    cos_angle = cp.cos(tail_angle)
    sin_angle = cp.sin(tail_angle)
    dx = u * cos_angle - v * sin_angle
    dy = u * sin_angle + v * cos_angle
    x = cp.rint(head_center[0] + dx).astype(cp.int32)
    y = cp.rint(head_center[1] + dy).astype(cp.int32)
    H, W = heatmap.shape
    mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    cp.add.at(heatmap, (y[mask], x[mask]), weight[mask])
    
    return heatmap

def generate_motion(multi_targets, inputs, base_kernel_length=1, length_scale=3, 
                            kernel_width=21, alpha=0.1, sigma=3, brightness_factor=0.1, motion_threshold=0.1):
    # multi_targets是一个长度为batch_size的列表，每个元素对应一个视频序列的所有帧标注
    batch_size = len(multi_targets) 
    image_h, image_w = 512, 512
    heatmaps = []
    num_frames = 5

    for i in range(batch_size):
        heatmap = cp.zeros((image_h, image_w), dtype=cp.float32)
        boxes_frames = multi_targets[i] # 获取第i个序列的帧标注
        first_targets = boxes_frames[0] 
        last_targets = boxes_frames[-1] # 以第一帧和最后一帧作为运动起点和终点
        if first_targets.shape[0] == 0:
            heatmaps.append(cp.asnumpy(heatmap))
            continue # 如果第一帧没有目标，直接把一个全0热力图放到目标heatmap里

        num_targets = first_targets.shape[0] # 第一帧目标个数
        for t in range(num_targets):
            initial = first_targets[t] # 遍历第一帧中每个目标
            initial_conv = initial 
            if last_targets.shape[0] == 0:
                final_conv = initial_conv # 如果最后一帧没有目标，直接把第一帧的目标赋给它
            else:
                if last_targets.shape[0] == num_targets:
                    final = last_targets[t]
                    final_conv = final # 如果第一帧最后一帧目标数相同，直接一一对应
                else:
                    centers = np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in last_targets])
                    initial_center = initial_conv[0:2]
                    distances = np.linalg.norm(centers - initial_center, axis=1)
                    min_index = np.argmin(distances)
                    final = last_targets[min_index]
                    final_conv = final # 计算目标中心欧几里得距离匹配最近目标
            
            x0, y0 = initial_conv[0], initial_conv[1]
            xf, yf = final_conv[0], final_conv[1]
            dx = xf - x0
            dy = yf - y0
            displacement = np.sqrt(dx**2 + dy**2)
            motion_angle = np.arctan2(dy, dx) # 计算起点终点的位移和角度
            
            if displacement < motion_threshold:
                x_head, y_head = int(round(xf)), int(round(yf))
                if 0 <= x_head < image_w and 0 <= y_head < image_h:
                    heatmap[y_head, x_head] += 1.0 # 位移太小视为静止，直接在终点位置强度+1
            else:
                dynamic_kernel_length = int(max(1, base_kernel_length + length_scale * displacement))
                brightness_scale = 1 + brightness_factor * displacement
                tail_angle = motion_angle + np.pi  
                head_center = (xf, yf)
                heatmap = add_comet_kernel(heatmap, head_center, tail_angle,
                                           kernel_length=dynamic_kernel_length,
                                           kernel_width=kernel_width, alpha=alpha, sigma=sigma,
                                           brightness_scale=brightness_scale)
        heatmaps.append(heatmap) # 将运动节点的彗星核加在运动热力图上
    return heatmaps 



class DBaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=False):
        super(DBaseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, 
                                      padding=(kernel_size - 1) // 2, 
                                      bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avgout = self.mlp(self.avg_pool(x))
        maxout = self.mlp(self.max_pool(x))
        scale_channel = self.sigmoid_channel(avgout + maxout)
        x = x * scale_channel  
        avgout_spatial = torch.mean(x, dim=1, keepdim=True)
        maxout_spatial, _ = torch.max(x, dim=1, keepdim=True)
        scale_spatial = torch.cat([avgout_spatial, maxout_spatial], dim=1)
        scale_spatial = self.sigmoid_spatial(self.conv_spatial(scale_spatial))
        x = x * scale_spatial
        return x

class Fusion_Module(nn.Module):
    def __init__(self, channels=[128, 256, 512], num_frame=5):
        super(Fusion_Module, self).__init__()
        self.k_conv = BaseConv(channels[0], channels[0] * 2, 3, 1, 1)
        self.fusion_conv = BaseConv(channels[0] * 4, channels[0] * 2, 3, 1, 1)
        self.cbam = CBAMBlock(channels[0] * 4, reduction=16, kernel_size=7)
        mid_channels = channels[0] * 2  # 256
        self.pre_conv = BaseConv(channels[0] * 4, mid_channels, 1, 1)
        self.upsample_branch = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            BaseConv(mid_channels, mid_channels, 3, 1, 1)
        )
        self.dilated1 = DBaseConv(mid_channels, mid_channels, 3, 1, dilation=1, padding=1)
        self.dilated2 = DBaseConv(mid_channels, mid_channels, 3, 1, dilation=2, padding=2)
        self.dilated3 = DBaseConv(mid_channels, mid_channels, 3, 1, dilation=4, padding=4)
        self.dilated_fuse = BaseConv(mid_channels * 3, mid_channels, 1, 1)
        self.merge_scale = BaseConv(mid_channels * 2, channels[0] * 2, 3, 1)
        self.residual_conv = nn.Sequential(
            BaseConv(channels[0] * 4, channels[0] * 2, 1, 1),
            nn.BatchNorm2d(channels[0] * 2)
        )
        
    def forward(self, motion, k_feat):
        k_feat_trans = self.k_conv(k_feat)
        fused = torch.cat([motion, k_feat_trans], dim=1) # motion特征与视觉特征在通道维度对齐
        att = self.cbam(fused) # CBAM自注意力加权融合特征
        pre_feat = self.pre_conv(att) # 通道压缩
        up_feat = self.upsample_branch(pre_feat) # 上采样
        d1 = self.dilated1(pre_feat) 
        d2 = self.dilated2(pre_feat)  
        d3 = self.dilated3(pre_feat) 
        dilated_out = torch.cat([d1, d2, d3], dim=1) 
        dilated_out = self.dilated_fuse(dilated_out) # 三种不同空洞率卷积，提取多尺度信息再融合
        up_feat_down = F.interpolate(up_feat, size=pre_feat.shape[2:], mode='bilinear', align_corners=False)
        multi_scale_fused = torch.cat([dilated_out, up_feat_down], dim=1)
        multi_scale_fused = self.merge_scale(multi_scale_fused) # 把上采样特征和dilated融合特征拼接融合          
        fused_out = self.fusion_conv(att) 
        fused_out_final = fused_out + multi_scale_fused
        fused_res = fused_out_final + self.residual_conv(fused)
        return [fused_res]




    

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [16, 32, 64], act = "silu"):
        super().__init__()
        Conv            =  BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            
            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        
        outputs = []
        for k, x in enumerate(inputs):
            x       = self.stems[k](x)
            cls_feat    = self.cls_convs[k](x)
            cls_output  = self.cls_preds[k](cls_feat)
            reg_feat    = self.reg_convs[k](x)
            reg_output  = self.reg_preds[k](reg_feat)
            obj_output  = self.obj_preds[k](reg_feat)
            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs



