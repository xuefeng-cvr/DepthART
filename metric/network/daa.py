import math
import torch
import torch.nn as nn
from .attention import AttentionBlock

class DAAStage(nn.Module):
    """
    Distortion-Aware Adapter per stage (parallel residual).
    Uses low-rank query (A,B) to attend to image tokens and camera embeddings.
    """

    def __init__(self, channels: int, cam_dim: int = 256, num_queries: int = 64, rank: int = 8, fusion_method: str = 'concat'):
        super().__init__()
        self.A = nn.Parameter(torch.randn(num_queries, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, channels) * 0.01)
        # depthwise + pointwise conv projections
        self.proj_cam_dw = nn.Conv2d(cam_dim, cam_dim, kernel_size=3, padding=1, groups=cam_dim, bias=False)
        self.proj_cam_pw = nn.Conv2d(cam_dim, channels, kernel_size=1, bias=False)
        self.proj_out_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.proj_out_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # self.proj_concat_out_dw = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.proj_concat_out_pw = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj_out_pw.weight)
        nn.init.zeros_(self.proj_out_pw.bias)
        self.fusion_method = fusion_method
        # 跨注意力块 - Q来自图像特征，K,V来自rays  
        self.cross_attention = AttentionBlock(  
            dim=channels,  
            num_heads=1,  # 单头注意力
            expansion=4,  
            dropout=0.0,  
            layer_scale=1e-5,  
            context_dim=channels,  # 启用跨注意力  
        )  # 直接使用跨注意力机制融合的话，容易把预训练提取特征的能力给打乱呢？

    def forward(self, feat: torch.Tensor, cam: torch.Tensor):
        """
        feat: (B, C, H, W)
        cam:  (1 or B, CE, H, W)
        """
        B, C, H, W = feat.shape
        N = H * W

        cam = cam.expand(B, -1, -1, -1)
        cam_proj = self.proj_cam_pw(self.proj_cam_dw(cam))  # (B, C, H, W)
        cam_tokens = cam_proj.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, C)

        if self.fusion_method == 'cross_attention':
             # 跨注意力：Q来自图像，K,V来自rays embeddings  （感觉可以在做cross_attention时做轻量化处理，尤其是在高分辨率下，逐像素token太耗时间了）
            attended_tokens = self.cross_attention(  
                tokens,             # Query  
                context=cam_tokens  # Key, Value  
            )  
            attended_features = attended_tokens.transpose(1, 2).reshape(B, C, H, W)
            # 后面接入深度可分离卷积投影前最好做层归一化
            # feat = feat + attended_features 
            # feat = self.proj_out_pw(self.proj_out_dw(attended_features))
            return attended_features
        elif self.fusion_method == 'additive':
            delta_feat = tokens + cam_tokens  # (B, N, C)
            delta_feat = delta_feat.transpose(1, 2).reshape(B, C, H, W)
            # delta_feat = self.proj_out_pw(self.proj_out_dw(delta_feat))
            # feat = feat + delta_feat
            return delta_feat
        elif self.fusion_method == 'concat':
            delta_feat = torch.cat([tokens, cam_tokens], dim=-1)  # (B, N, 2C)
            delta_feat = delta_feat.transpose(1, 2).reshape(B, 2 * C, H, W)
            delta_feat = self.proj_concat_out_pw(delta_feat)
            feat = feat + delta_feat # 残差不合理吧。。。
        else:
            q = self.A @ self.B  # (Q, C)
            k_feat = tokens.transpose(1, 2)  # (B, C, N)
            k_cam = cam_tokens.transpose(1, 2)  # (B, C, N)

            att_f = torch.softmax((q @ k_feat) / math.sqrt(C), dim=-1)  # (B, Q, N)
            att_c = torch.softmax((q @ k_cam) / math.sqrt(C), dim=-1)
            att = 0.5 * (att_f + att_c)

            delta = att.transpose(1, 2) @ q  # (B, N, C)
            delta = delta.transpose(1, 2).reshape(B, C, H, W)
            delta = self.proj_out_pw(self.proj_out_dw(delta))
            feat = feat + delta

        return feat
# import math
# import torch
# import torch.nn as nn
# from .attention import AttentionBlock

# class DAAStage(nn.Module):
#     """
#     Distortion-Aware Adapter per stage (parallel residual).
#     Uses low-rank query (A,B) to attend to image tokens and camera embeddings.
#     """

#     def __init__(self, channels: int, cam_dim: int = 256, num_queries: int = 64, rank: int = 8, cross_attention_fusion: bool = True):
#         super().__init__()
#         self.A = nn.Parameter(torch.randn(num_queries, rank) * 0.01)
#         self.B = nn.Parameter(torch.randn(rank, channels) * 0.01)
#         # depthwise + pointwise conv projections
#         self.proj_cam_dw = nn.Conv2d(cam_dim, cam_dim, kernel_size=3, padding=1, groups=cam_dim, bias=False)
#         self.proj_cam_pw = nn.Conv2d(cam_dim, channels, kernel_size=1, bias=False)
#         self.proj_out_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
#         self.proj_out_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
#         # nn.init.zeros_(self.proj_out_pw.weight)
#         # nn.init.zeros_(self.proj_out_pw.bias)
#         self.cross_attention_fusion = cross_attention_fusion
#         # 跨注意力块 - Q来自图像特征，K,V来自rays  
#         self.cross_attention = AttentionBlock(  
#             dim=channels,  
#             num_heads=1,  # 单头注意力
#             expansion=4,  
#             dropout=0.0,  
#             layer_scale=-1.0,  
#             context_dim=channels,  # 启用跨注意力  
#         )  

#     def forward(self, feat: torch.Tensor, cam: torch.Tensor):
#         """
#         feat: (B, C, H, W)
#         cam:  (1 or B, CE, H, W)
#         """
#         B, C, H, W = feat.shape
#         N = H * W

#         cam = cam.expand(B, -1, -1, -1)
#         cam_proj = self.proj_cam_pw(self.proj_cam_dw(cam))  # (B, C, H, W)
#         cam_tokens = cam_proj.flatten(2).transpose(1, 2)  # (B, N, C)
#         tokens = feat.flatten(2).transpose(1, 2)  # (B, N, C)

#         if self.cross_attention_fusion:
#              # 跨注意力：Q来自图像，K,V来自rays embeddings 
#             attended_features = self.cross_attention(  
#                 tokens,             # Query  
#                 context=cam_tokens  # Key, Value  
#             )  
#             attended_features = attended_features.transpose(1, 2).reshape(B, C, H, W)
#             # attended_features = feat + self.proj_out_pw(self.proj_out_dw(attended_features))
#             return attended_features
#         else:
#             q = self.A @ self.B  # (Q, C)
#             k_feat = tokens.transpose(1, 2)  # (B, C, N)
#             k_cam = cam_tokens.transpose(1, 2)  # (B, C, N)

#             att_f = torch.softmax((q @ k_feat) / math.sqrt(C), dim=-1)  # (B, Q, N)
#             att_c = torch.softmax((q @ k_cam) / math.sqrt(C), dim=-1)
#             att = 0.5 * (att_f + att_c)

#             delta = att.transpose(1, 2) @ q  # (B, N, C)
#             delta = delta.transpose(1, 2).reshape(B, C, H, W)
#             delta = self.proj_out_pw(self.proj_out_dw(delta))
#             feat = feat + delta

#         return feat