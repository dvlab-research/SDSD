import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
try:
    from models.archs.dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError as e:
    print(e)
    raise ImportError('Failed to import DCNv2 module.')


class PAL(nn.Module):
    def __init__(self, nf=64, groups=8):
        super(PAL, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                               extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack([nbr_fea_l[2], L3_offset]))

        # L2
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))

        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack([nbr_fea_l[0], L1_offset])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))

        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack([L1_fea, offset]))

        return L1_fea


class Fusion(nn.Module):
    def __init__(self, nf=64, nframes=5, center=2):
        super(Fusion, self).__init__()
        self.center = center
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames

        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))

        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea


class denoise_illumination_enhance(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=True, w_TSA=True):
        super(denoise_illumination_enhance, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.pcd_align = PAL(nf=nf, groups=groups)
        if self.w_TSA:
            self.tsa_fusion = Fusion(nf=nf, nframes=nframes, center=self.center)
        else:
            self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        ### branch for illumination enhancement
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### branch for denoising
        self.recon_trunk_denoise = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        self.upconv1_denoise = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2_denoise = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle_denoise = nn.PixelShuffle(2)
        self.HRconv_denoise = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last_denoise = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        L1_fea = self.lrelu(self.conv_first_1(x.view(-1, C, H, W)))
        L1_fea = self.lrelu(self.conv_first_2(L1_fea))
        L1_fea = self.lrelu(self.conv_first_3(L1_fea))
        H, W = H // 4, W // 4

        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)


        ref_fea_l = [L1_fea[:, self.center, :, :, :].clone(),
                     L2_fea[:, self.center, :, :, :].clone(),
                     L3_fea[:, self.center, :, :, :].clone()]

        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]

        if not self.w_TSA:
            aligned_fea = aligned_fea.view(B, -1, H, W)
        fea = self.tsa_fusion(aligned_fea)
        ##########################################################################################

        ## the branch for denoising
        out_noise = self.recon_trunk_denoise(fea)
        out_noise = self.lrelu(self.pixel_shuffle_denoise(self.upconv1_denoise(out_noise)))
        out_noise = self.lrelu(self.pixel_shuffle_denoise(self.upconv2_denoise(out_noise)))
        out_noise = self.lrelu(self.HRconv_denoise(out_noise))
        out_noise = self.conv_last_denoise(out_noise)
        base = x_center
        base_denoise = base - out_noise


        ## the branch for learning illumination map
        out_ill = self.recon_trunk(fea)
        out_ill = self.lrelu(self.pixel_shuffle(self.upconv1(out_ill)))
        out_ill = self.lrelu(self.pixel_shuffle(self.upconv2(out_ill)))
        out_ill = self.lrelu(self.HRconv(out_ill))
        out_ill = self.conv_last(out_ill)
        out_ill = torch.min(torch.max(out_ill, base), torch.ones_like(out_ill))
        base_ill = torch.div(base_denoise, torch.clamp(out_ill, 0.001, 1))
        return base_ill, out_ill, base_denoise

