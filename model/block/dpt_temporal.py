import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

from .motion_module import TemporalModule
from .dpt import DPT

class DPTTemporal(DPT):
    def __init__(
        self,
        in_channels,
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        num_heads=8,
        num_transformer_block=1,
        num_attention_blocks=2,
    ):
        super().__init__(
            in_channels=in_channels,
            features=features,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
        )

        mm_kwargs = EasyDict(
            num_attention_heads = num_heads,
            num_transformer_block = num_transformer_block,
            num_attention_blocks = num_attention_blocks,
            temporal_max_len = num_frames,
            zero_initialize = True,
            pos_embedding_type = pe,
        )

        self.tm_l3 = TemporalModule(in_channels=out_channels[2], **mm_kwargs)
        self.tm_l4 = TemporalModule(in_channels=out_channels[3], **mm_kwargs)
        self.tm_p4 = TemporalModule(in_channels=features, **mm_kwargs)
        self.tm_p3 = TemporalModule(in_channels=features, **mm_kwargs)

    @staticmethod
    def _apply_temporal(mod, feat_btchw, B, T, cached_slice=None):
        """
        feat_btchw: [B*T, C, H, W] → TemporalModule([B,T,C,H,W]) → [B*T, C, H, W]
        """
        x = feat_btchw.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)  # [B,T,C,H,W]
        x, _h = mod(x, None, None, cached_slice)
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous()
        return x

    def forward(
        self,
        out_features,
        image,
        patch_h, patch_w,
        pos,
        frame_length: int,
    ):
        outs = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x_tok, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x_tok)
                x_tok = self.readout_projects[i](torch.cat((x_tok, readout), -1))
            else:
                x_tok = x[0]
            x_tok = x_tok.permute(0, 2, 1).reshape((x_tok.shape[0], x_tok.shape[-1], patch_h, patch_w))
            x_tok = self.projects[i](x_tok)
            x_tok = self.resize_layers[i](x_tok)
            outs.append(x_tok)

        layer_1, layer_2, layer_3, layer_4 = outs
        B = layer_1.shape[0] // frame_length
        T = frame_length


        pos = self.pos_bridge(pos)

        layer_3 = self._apply_temporal(self.tm_l3, layer_3, B, T)
        layer_4 = self._apply_temporal(self.tm_l4, layer_4, B, T)

        img_f4, img_f3, img_f2, img_f1 = self.image_encoder(image)

        p4 = F.interpolate(pos, size=img_f4.shape[-2:], mode='bilinear', align_corners=True)
        p3 = F.interpolate(pos, size=img_f3.shape[-2:], mode='bilinear', align_corners=True)
        p2 = F.interpolate(pos, size=img_f2.shape[-2:], mode='bilinear', align_corners=True)
        p1 = F.interpolate(pos, size=img_f1.shape[-2:], mode='bilinear', align_corners=True)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(prev_feature=None, current_feature=layer_4_rn, image_feature=img_f4, pos=p4)
        path_4 = self._apply_temporal(self.tm_p4, path_4, B, T)

        path_3 = self.scratch.refinenet3(prev_feature=path_4, current_feature=layer_3_rn, image_feature=img_f3, pos=p3)
        path_3 = self._apply_temporal(self.tm_p3, path_3, B, T)

        path_2 = self.scratch.refinenet2(prev_feature=path_3, current_feature=layer_2_rn, image_feature=img_f2, pos=p2)
        path_1 = self.scratch.refinenet1(prev_feature=path_2, current_feature=layer_1_rn, image_feature=img_f1, pos=p1)

        return path_1