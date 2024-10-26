import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import warp
from .loss import *
from .fusion_2E import Fusion_Net
from .network_utils import *

def tonemap(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)

def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )

def cur_tone_perturb(cur, test_mode, d=0.7):
    if not test_mode:
        b, c, h, w = cur.shape
        gamma_aug = torch.exp(torch.rand(b, 3, 1, 1) * 2 * d - d)
        gamma_aug = gamma_aug.to(cur.device)
        cur_aug = torch.pow(cur, 1.0 / gamma_aug)
    else:
        cur_aug = cur
    return cur_aug

class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :].clone())
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :].clone())
        out = self.prelu(x + self.conv5(out))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(3, 32, 3, 2, 1), 
            convrelu(32, 32, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(32, 48, 3, 2, 1), 
            convrelu(48, 48, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(48, 72, 3, 2, 1), 
            convrelu(72, 72, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(72, 96, 3, 2, 1), 
            convrelu(96, 96, 3, 1, 1)
        )
        
    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4

class IPdecoder3(nn.Module):
    def __init__(self):
        super(IPdecoder3, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(72+72, 144), 
            convrelu(144, 72)
        )
        self.up = nn.ConvTranspose2d(72, 48, 4, 2, 1, bias=True)
    def forward(self, warped_f0, warped_f1):
        inp = torch.cat([warped_f0, warped_f1], 1)
        IP_out = self.convblock(inp)
        IP_out_up = self.up(IP_out)
        return IP_out, IP_out_up

class IPdecoder2(nn.Module):
    def __init__(self):
        super(IPdecoder2, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(48+48+48, 96), 
            convrelu(96, 48)
        )
        self.up = nn.ConvTranspose2d(48, 32, 4, 2, 1, bias=True)
    def forward(self, warped_f0, warped_f1, f_ip_up_0):
        inp = torch.cat([warped_f0, warped_f1, f_ip_up_0], 1)
        IP_out = self.convblock(inp)
        IP_out_up = self.up(IP_out)
        return IP_out, IP_out_up

class IPdecoder1(nn.Module):
    def __init__(self):
        super(IPdecoder1, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(32+32+32, 64), 
            convrelu(64, 32)
        )
        self.up = nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=True)
    def forward(self, warped_f0, warped_f1, f_ip_up_0):
        inp = torch.cat([warped_f0, warped_f1, f_ip_up_0], 1)
        IP_out = self.convblock(inp)
        IP_out_up = self.up(IP_out)
        return IP_out, IP_out_up

class IPdecoder0(nn.Module):
    def __init__(self):
        super(IPdecoder0, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(6+3, 32), 
            convrelu(32, 3)
        )
    def forward(self, warped_f0, warped_f1, f_ip_up_0):
        inp = torch.cat([warped_f0, warped_f1, f_ip_up_0], 1)
        IP_out = self.convblock(inp)
        return IP_out

class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(192+96, 192), 
            ResBlock(192, 32), 
            nn.ConvTranspose2d(192, 76, 4, 2, 1, bias=True)
        )
        
    def forward(self, f0, f1, fc):
        f_in = torch.cat([f0, f1, fc], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(220+72+72, 216), 
            ResBlock(216, 32), 
            nn.ConvTranspose2d(216, 52, 4, 2, 1, bias=True)
        )
        self.ipdecoder = IPdecoder3()

    def forward(self, ft_, f0, f1, fc, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_ip, f_ip_up = self.ipdecoder(f0_warp, f1_warp)
        f_in = torch.cat([ft_, fc, f_ip, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out, f_ip, f_ip_up


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(148+48+48, 144), 
            ResBlock(144, 32), 
            nn.ConvTranspose2d(144, 36, 4, 2, 1, bias=True)
        )
        self.ipdecoder = IPdecoder2()

    def forward(self, ft_, f0, f1, fc, up_flow0, up_flow1, f_ip_up_0):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_ip, f_ip_up = self.ipdecoder(f0_warp, f1_warp, f_ip_up_0)
        f_in = torch.cat([ft_, fc, f_ip, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out, f_ip, f_ip_up


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(100+32+32, 96), 
            ResBlock(96, 32), 
            nn.ConvTranspose2d(96, 7, 4, 2, 1, bias=True)
        )
        self.ipdecoder = IPdecoder1()

    def forward(self, ft_, f0, f1, fc, up_flow0, up_flow1, f_ip_up_0):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_ip, f_ip_up = self.ipdecoder(f0_warp, f1_warp, f_ip_up_0)
        f_in = torch.cat([ft_, fc, f_ip, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out, f_ip, f_ip_up

class Decoder0(nn.Module):
    def __init__(self):
        super(Decoder0, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(15, 32), 
            ResBlock(32, 16), 
            convrelu(32, 3)
        )
        self.ipdecoder = IPdecoder0()
        
    def forward(self, hdr, f0, f1, fc, up_flow0, up_flow1, expos):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_ip = self.ipdecoder(f0_warp, f1_warp)
        f_ip_hdr = ldr_to_hdr(f_ip, expos[0])
        fc_hdr = ldr_to_hdr(fc, expos[1])
        f_in = torch.cat([hdr, fc, fc_hdr, f_ip, f_ip_hdr], 1)
        f_out = self.convblock(f_in)
        return f_out, f_ip



class MSANet_3E(nn.Module):
    def __init__(self, local_rank=-1, lr=1e-4):
        super(MSANet_3E, self).__init__()
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()
        self.decoder0 = Decoder0()
        self.ipdecoder0 = IPdecoder0()
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)
        self.fusion_net = Fusion_Net(39, 7)
        self.merge_hdr = MergeHDRModule()


    def forward(self, ldrs, hdrs, rev1_ldrs, expos, test_mode = False):

        pt_img_c = cur_tone_perturb(ldrs[2], test_mode)
        # pt_img_c = ldrs[1]
        f0_1, f0_2, f0_3, f0_4 = self.encoder(ldrs[0])
        f1_1, f1_2, f1_3, f1_4 = self.encoder(ldrs[1])
        
        f3_1, f3_2, f3_3, f3_4 = self.encoder(ldrs[3])
        f4_1, f4_2, f4_3, f4_4 = self.encoder(ldrs[4])

        fc_1, fc_2, fc_3, fc_4 = self.encoder(pt_img_c)
        if not test_mode:
            ft_1, ft_2, ft_3, ft_4 = self.encoder(hdrs[2])
            fr1_1, fr1_2, fr1_3, fr1_4 = self.encoder(rev1_ldrs[2])
            fr2_1, fr2_2, fr2_3, fr2_4 = self.encoder(rev2_ldrs[2])



        out4 = self.decoder4(f0_4, f3_4, fc_4, f1_4, f4_4)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]
        

        out3, f_ip_3, f_ip_3_up = self.decoder3(ft_3_, f0_3, f1_3, fc_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        out2, f_ip_2, f_ip_2_up = self.decoder2(ft_2_, f0_2, f1_2, fc_2, up_flow0_3, up_flow1_3, f_ip_3_up)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_  = out2[:, 4:]

        out1, f_ip_1, f_ip_1_up = self.decoder1(ft_1_, f0_1, f1_1, fc_1, up_flow0_2, up_flow1_2, f_ip_2_up)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        pre_hdr = out1[:, 4:]

        img_p_warp = warp(ldrs[0], up_flow0_1)
        img_n_warp = warp(ldrs[2], up_flow1_1)
        img_ip = self.ipdecoder0(img_p_warp, img_n_warp, f_ip_1_up)
        # ldrs_inp = torch.cat([pt_img_c, img_p_warp, img_n_warp, ldrs[0], ldrs[2] ], 1)
        ldrs_inp = torch.cat([pt_img_c, img_ip, img_p_warp, img_n_warp, ldrs[0], ldrs[2]], 1)
        hdr_c = ldr_to_hdr(pt_img_c, expos[1])
        hdr_p = ldr_to_hdr(ldrs[0], expos[0])
        hdr_n = ldr_to_hdr(ldrs[2], expos[2])
        warp_hdr_p = ldr_to_hdr(img_p_warp, expos[0])
        warp_hdr_n = ldr_to_hdr(img_n_warp, expos[2])
        hdr_ip = ldr_to_hdr(img_ip, expos[0])
        hdrs_inp = torch.cat([pre_hdr, hdr_c, hdr_ip, warp_hdr_p, warp_hdr_n, hdr_p, hdr_n], 1)
        fusion_hdr = [pre_hdr, hdr_c, hdr_ip, warp_hdr_p, warp_hdr_n, hdr_p, hdr_n]
        fusion_inp = torch.cat([ldrs_inp, hdrs_inp], 1)
        # hdrs_inp = torch.cat([pre_hdr, warp_hdr_p, warp_hdr_n, hdr_p, hdr_n], 1)
        # fusion_hdr = [pre_hdr, pt_img_c, warp_hdr_p, warp_hdr_n, hdr_p, hdr_n]
        # fusion_inp = torch.cat([ldrs_inp, hdrs_inp], 1)
        imgt_pred = self.fusion_net(fusion_inp, fusion_hdr)

        # imgt_pred, img_ip = self.decoder0(pre_hdr, ldrs[0], ldrs[2], pt_img_c, up_flow0_1, up_flow1_1, expos)
        img_ip = torch.clamp(img_ip, 0, 1)
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        if not test_mode:
            imgt_pred_tm = tonemap(imgt_pred)
            imgt_tm = tonemap(hdrs[1])
            loss_rec = self.l1_loss(imgt_pred_tm - imgt_tm) + self.tr_loss(imgt_pred_tm, imgt_tm) + self.l1_loss(img_ip, rev_ldrs[1])
            loss_geo = (self.gc_loss(ft_1_, ft_1) + self.gc_loss(ft_2_, ft_2) + self.gc_loss(ft_3_, ft_3)
            + self.gc_loss(f_ip_1, fr_1) + self.gc_loss(f_ip_2, fr_2) + self.gc_loss(f_ip_3, fr_3))

            loss_msa = (Muti_scale_hdraignloss(ldrs[1], hdrs, up_flow0_1, up_flow1_1)
                    +  Muti_scale_hdraignloss(ldrs[1], hdrs, 2.0 * resize(up_flow0_2, 2.0), 2.0 * resize(up_flow1_2, 2.0))
                    +  Muti_scale_hdraignloss(ldrs[1], hdrs, 4.0 * resize(up_flow0_3, 4.0), 4.0 * resize(up_flow1_3, 4.0))
                    +  Muti_scale_hdraignloss(ldrs[1], hdrs, 8.0 * resize(up_flow0_4, 8.0), 8.0 * resize(up_flow1_4, 8.0)))

            return imgt_pred, loss_rec, loss_geo, loss_msa
        else:
            return imgt_pred
