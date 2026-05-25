import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=True)

    def forward(self, x):
        w = self.pool(x)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w

class RefineBlock(nn.Module):
    def __init__(self, in_features, out_features, image_features, pos_channels):
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(in_features)
        self.resConfUnit2 = ResidualConvUnit(in_features)
        
        fuse_in_channels = in_features + image_features

        self.pos = nn.Conv2d(pos_channels, image_features, kernel_size=1, stride=1, padding=0, bias=False)

        self.chan_att = ChannelAttention(fuse_in_channels, reduction=16)

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fuse_in_channels, out_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )
        
        self.out_conv = nn.Conv2d(out_features, out_features, kernel_size=1, stride=1, padding=0)

    def forward(self, prev_feature, current_feature, image_feature, pos):
        if prev_feature is not None:
            prev_feature = F.interpolate(prev_feature, size=current_feature.shape[-2:], mode='bilinear', align_corners=True)
            current_feature = prev_feature + current_feature
            current_feature = self.resConfUnit1(current_feature)
        
        current_feature = self.resConfUnit2(current_feature)
        
        current_feature = F.interpolate(current_feature, size=image_feature.shape[-2:], mode='bilinear', align_corners=True)

        pos = self.pos(pos)
        fused = torch.cat([current_feature, image_feature + pos], dim=1)
        fused = self.chan_att(fused)
        fused = self.fuse_conv(fused)
        out = self.out_conv(fused)
        return out
    
class CSEPosBridge(nn.Module):
    def __init__(self, in_ch=16, out_ch=16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, P):
        P = self.proj(P)
        P = F.relu(P, True)
        return P

class ImageEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class ImageEncoder(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.b1 = ImageEncoderBlock(3, features, 2)  # f1
        self.b2 = ImageEncoderBlock(features, features, 2)  # f2
        self.b3 = ImageEncoderBlock(features, features, 2)  # f3
        self.b4 = ImageEncoderBlock(features, features, 2)  # f4

    def forward(self, x):
        f1 = self.b1(x)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)
        return [f4, f3, f2, f1]

class DPT(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256,
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPT, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
                
        self.image_encoder = ImageEncoder(features)

        self.pos_channels = 16
        self.pos_bridge = CSEPosBridge(self.pos_channels, self.pos_channels)
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.refinenet1 = RefineBlock(features, features, features, self.pos_channels)
        self.scratch.refinenet2 = RefineBlock(features, features, features, self.pos_channels)
        self.scratch.refinenet3 = RefineBlock(features, features, features, self.pos_channels)
        self.scratch.refinenet4 = RefineBlock(features, features, features, self.pos_channels)
    
    def forward(self, out_features, image, patch_h, patch_w, pos):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        img_f4, img_f3, img_f2, img_f1 = self.image_encoder(image)
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        pos = self.pos_bridge(pos)
        p4 = F.interpolate(pos, size=img_f4.shape[-2:], mode='bilinear', align_corners=True)
        p3 = F.interpolate(pos, size=img_f3.shape[-2:], mode='bilinear', align_corners=True)
        p2 = F.interpolate(pos, size=img_f2.shape[-2:], mode='bilinear', align_corners=True)
        p1 = F.interpolate(pos, size=img_f1.shape[-2:], mode='bilinear', align_corners=True)

        
        path_4 = self.scratch.refinenet4(prev_feature=None, current_feature=layer_4_rn, image_feature=img_f4, pos=p4)
        path_3 = self.scratch.refinenet3(prev_feature=path_4, current_feature=layer_3_rn, image_feature=img_f3, pos=p3)
        path_2 = self.scratch.refinenet2(prev_feature=path_3, current_feature=layer_2_rn, image_feature=img_f2, pos=p2)
        path_1 = self.scratch.refinenet1(prev_feature=path_2, current_feature=layer_1_rn, image_feature=img_f1, pos=p1)
        
        return path_1

class Head(nn.Module):
    def __init__(self, features, out_channels):
        super().__init__()
        head_features = 32
        
        self.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, head_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features, out_channels, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x, ori_h, ori_w):
        x = self.output_conv1(x)
        x = F.interpolate(x, (int(ori_h), int(ori_w)), mode="bilinear", align_corners=True)
        x = self.output_conv2(x)
        return x
    

