import torch
import torch.nn as nn
from einops import rearrange


class ScaleFormerHead(nn.Module):
    """
    Lightweight transformer decoder predicting global scale from deepest features + camera embedding.
    """

    def __init__(
        self,
        in_dim: int,
        cam_dim: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        num_queries: int = 8,
        ff_ratio: int = 4,
    ):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))
        self.proj_ctx = nn.Linear(in_dim + cam_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_ratio,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

    def forward(self, feat: torch.Tensor, cam: torch.Tensor):
        """
        feat: (B, C, H, W) deepest stage feature.
        cam:  (1 or B, CE, H, W) camera embedding at same scale.
        """
        B, C, H, W = feat.shape
        cam = cam.expand(B, -1, -1, -1)
        tokens = torch.cat([feat, cam], dim=1)  # (B, C+CE, H, W)
        tokens = rearrange(tokens, "b c h w -> b (h w) c")
        ctx = self.proj_ctx(tokens)  # (B, N, d_model)

        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(tgt=queries, memory=ctx)  # (B, Q, d_model)
        scales = torch.sigmoid(self.proj_out(decoded))  # (B, Q, 1)
        return scales.mean(dim=1)  # (B, 1)
