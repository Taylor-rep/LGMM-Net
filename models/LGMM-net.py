import torch
from torch import nn
import torch.nn.functional as F
from timm.layers.weight_init import trunc_normal_
import math
from mamba_ssm import Mamba
from einops.layers.torch import Rearrange

class SF_Mamba(nn.Module):
    def __init__(self, in_channels, out_channels, d_state=16, d_conv=4, expand=2, num_branches=3):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k//2, groups=in_channels)
            for k in [3, 5, 7][:num_branches]
        ])
        self.pw_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
            for _ in range(num_branches)
        ])
        self.mambas = nn.ModuleList([
            Mamba(
                d_model=in_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            for _ in range(num_branches)
        ])
        self.fuse = nn.Conv2d(in_channels * num_branches, out_channels, kernel_size=1)
        self.gate = nn.Parameter(torch.ones(num_branches))
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x.flatten(2).transpose(1,2)).transpose(1,2).reshape(B, C, H, W)
        outs = []
        for i in range(len(self.dw_convs)):
            out = self.dw_convs[i](x_norm)
            out = self.pw_convs[i](out)
            out_ = out.view(B, C, H*W).transpose(1,2)  # (B, seq, C)
            out_ = self.mambas[i](out_)
            out_ = out_.transpose(1,2).view(B, C, H, W)
            outs.append(out_ * self.gate[i])
        out = torch.cat(outs, dim=1)
        out = self.fuse(out)
        out = out + self.proj(x) * self.skip_scale
        return out

class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim//4, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)

        return att1, att2, att3, att4, att5


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                          nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_




class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class BRM(nn.Module):
    def __init__(self, in_channels, dilation=1):
        super(BRM, self).__init__()
        assert in_channels % 2 == 0, "BRM通道数必须能被2整除"
        self.split = in_channels // 2

        # 左分支：通道注意力 + 异形卷积
        self.ca = ChannelAttention(self.split)
        self.l_conv1 = nn.Conv2d(self.split, self.split, kernel_size=(1, 3), padding=(0, 1), dilation=dilation)
        self.l_conv2 = nn.Conv2d(self.split, self.split, kernel_size=(3, 1), padding=(1, 0), dilation=dilation)
        self.l_bn = nn.BatchNorm2d(self.split)

        # 右分支：空间注意力 + 异形卷积
        self.sa = SpatialAttention()
        self.r_conv1 = nn.Conv2d(self.split, self.split, kernel_size=(1, 3), padding=(0, 1), dilation=dilation)
        self.r_conv2 = nn.Conv2d(self.split, self.split, kernel_size=(3, 1), padding=(1, 0), dilation=dilation)
        self.r_bn = nn.BatchNorm2d(self.split)

        # 跨分支信息融合
        self.inter_left = nn.Conv2d(self.split, self.split, 1)
        self.inter_right = nn.Conv2d(self.split, self.split, 1)

        # 输出融合
        self.out_bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def channel_shuffle(self, x, groups=2):
        B, C, H, W = x.shape
        channels_per_group = C // groups
        x = x.view(B, groups, channels_per_group, H, W)
        x = x.transpose(1, 2).contiguous()
        return x.view(B, C, H, W)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)

        # 左分支：通道注意力增强
        x1 = self.ca(x1) * x1
        x1 = self.l_conv1(x1)
        x1 = self.l_conv2(x1)
        x1 = self.l_bn(x1)

        # 右分支：空间注意力增强
        x2 = self.sa(x2) * x2
        x2 = self.r_conv1(x2)
        x2 = self.r_conv2(x2)
        x2 = self.r_bn(x2)

        # 跨通道交互
        x1 = x1 + self.inter_left(x2)
        x2 = x2 + self.inter_right(x1)

        # 融合
        out = torch.cat([x1, x2], dim=1)
        out = self.out_bn(out)
        out = out + x  # 残差
        out = self.channel_shuffle(out)
        return self.relu(out)
#ECA注意力模块
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class BFE_Mamba(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pvm = PVMLayer(input_dim, output_dim, d_state, d_conv, expand)
        self.brm = BRM(input_dim)
        self.brm_proj = nn.Conv2d(input_dim, output_dim, 1)
        self.fusion_weight = nn.Parameter(torch.ones(2))
        self.eca = ECA(output_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        pvm_out = self.pvm(x)
        brm_out = self.brm(x)
        brm_out = self.brm_proj(brm_out)
        weights = torch.softmax(self.fusion_weight, dim=0)
        fused = weights[0] * pvm_out + weights[1] * brm_out
        fused = self.fusion(fused)
        out = self.eca(fused)
        return out


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # 修正reduction，防止channels // reduction为0
        reduction = min(reduction, channels)
        if channels // reduction < 1:
            reduction = 1
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        # 空间注意力
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
    def forward(self, x):
        # 通道注意力
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_att
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        return x

class DSEB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lambda_edge = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.cbam = CBAM(channels)
    def forward(self, x):
        # Feature Edge Amplifier
        s0, s1, s2 = 1.0, 0.75, 0.5
        B, C, H, W = x.shape
        def scale_op(feat, scale):
            h, w = int(H * scale), int(W * scale)
            down = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)
            up = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=True)
            return up
        Fu1 = scale_op(x, s1)
        Fu2 = scale_op(x, s2)
        F_edge = torch.abs(Fu1 - Fu2)
        fea_out = x + self.lambda_edge * F_edge
        # Differential Attention (CBAM)
        diff_out = self.cbam(fea_out)
        return diff_out

class DEA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dseb = DSEB(channels)
    def forward(self, x):
        x = self.dseb(x)
        return x


class CGASpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn

class CGAChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

class CGAPixelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8, t6_dim=None):
        super().__init__()
        self.sa = CGASpatialAttention()
        self.ca = CGAChannelAttention(dim, reduction)
        self.pa = CGAPixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        # 新增：如果t6通道和dim不一致，做投影
        self.t6_proj = nn.Identity() if t6_dim is None or t6_dim == dim else nn.Conv2d(t6_dim, dim, 1)

    def forward(self, x, y):
        y_proj = self.t6_proj(y)
        initial = x + y_proj
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y_proj
        result = self.conv(result)
        return result


class MSR_Mamba(nn.Module):
    def __init__(self, c_list, agca_ratio=8):
        super().__init__()
        self.c_list = c_list
        self.cga = CGAFusion(c_list[-1], reduction=agca_ratio)  # 64通道
        self.sf_mamba = SF_Mamba(c_list[-1], c_list[-1])  # 64->64
        self.res_weight = nn.Parameter(torch.ones(1))  # 可学习残差权重
        self.norm = nn.GroupNorm(4, c_list[-1])         # 归一化
        self.act = nn.GELU()                            # 激活

    def forward(self, t6):  # 只接收encoder6的输出
        enhanced = self.cga(t6, t6)  # CGA注意力增强
        enhanced = self.sf_mamba(enhanced)  # SF-Mamba全局建模
        output = t6 + self.res_weight * enhanced  # 加权残差
        output = self.norm(output)                # 归一化
        output = self.act(output)                 # 激活
        return output

class LGMM_net(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64],
                split_att='fc', bridge=False):
        super().__init__()
        self.bridge = bridge
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            BFE_Mamba(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            BFE_Mamba(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            BFE_Mamba(input_dim=c_list[4], output_dim=c_list[5])
        )
        if bridge:
            self.bridge_block = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge Bridge used')
        self.sf_mamba = MSR_Mamba(c_list, agca_ratio=8)
        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        )
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        )
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.dea1 = DEA(c_list[0])
        self.dea2 = DEA(c_list[1])
        self.dea3 = DEA(c_list[2])
        self.dea4 = DEA(c_list[3])
        self.dea5 = DEA(c_list[4])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out # b, c4, H/32, W/32

        if self.bridge:
            t1, t2, t3, t4, t5 = self.bridge_block(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        # MSR_Mamba: 只处理encoder6的输出
        out = self.sf_mamba(out)
        out5 = F.gelu(self.dbn1(self.decoder1(out))) # b, c4, H/32, W/32
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        out5 = self.dea5(out5)

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        out4 = self.dea4(out4)

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        out3 = self.dea3(out3)

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4
        out2 = self.dea2(out2)

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        out1 = self.dea1(out1)

        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W

        return torch.sigmoid(out0)


