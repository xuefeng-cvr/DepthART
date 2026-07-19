import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util.blocks import FeatureFusionBlock, _make_scratch
import tinyvim.model.tinyvim  # 确保注册生效
from timm.models import create_model

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.conv_block(x)
    
class DPTHead(nn.Module):
    def __init__(
        self,
        features=48,                        # 统一通道数
        out_channels=[48, 64, 168, 224],    # backbone各层特征图的通道数
        use_bn=False, 
    ):
        super(DPTHead, self).__init__()
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 16
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )
    
    def forward(self, features, size_h, size_w):
    
        layer_1, layer_2, layer_3, layer_4 = features   # [b,48,56,56] [b,64,28,28] [b,168,14,14] [b,168,7,7] 
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)    # 统一调整到 48维
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])                # [b,48,7,7] -> [b,48,14,14]   
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])        # [b,48,14,14] -> [b,48,28,28]
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])        # [b,48,28,28] -> [b,48,56,56]
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)    # 特征融合到[b,48,56,56]        # [b,48,56,56] -> [b,48,112,112]
        
        out = self.scratch.output_conv1(path_1)                                                 # [b,48,112,112] -> [b,24,112,112]
        out = F.interpolate(out, (size_h, size_w), mode="bilinear", align_corners=True)         # [b,24,112,112] -> [b,24,224,224]
        out = self.scratch.output_conv2(out)                                                    # [b,24,224,224] -> [b,16,224,224] -> [b,1,224,224]
        
        return out


class TinyVimDepth(nn.Module):
    def __init__(
        self, 
        encoder="S", # 'S', 'B', 'L'  
        use_bn=False, 
    ):
        super(TinyVimDepth, self).__init__()
        
        if encoder=="S":
            """创建TinyViM_S模型"""
            self.pretrained = create_model(  
                            'TinyViM_S',  
                            num_classes=1000,  
                            pretrained=False,  
                            fork_feat=True    
                        )  
            self.features = 48
            self.out_channels = [48, 64, 168, 224]
        elif encoder=="B":
            """创建TinyViM_B模型"""
            self.pretrained = create_model(  
                            'TinyViM_B',  
                            num_classes=1000,  
                            pretrained=False,  
                            fork_feat=True    
                        )  
            self.features = 48
            self.out_channels = [48, 96, 192, 384]
        elif encoder=="L":
            """创建TinyViM_L模型"""
            self.pretrained = create_model(
                            'TinyViM_L',
                            num_classes=1000,
                            pretrained=False,
                            fork_feat=True 
                        )
            self.features = 64
            self.out_channels = [64, 128, 384, 512]
        self.depth_head = DPTHead(features=self.features,out_channels=self.out_channels,use_bn=use_bn)
    
    
    def forward(self, x):
        # 提取四层特征  [b,48,56,56]、[b, 64, 28, 28]、[b, 168, 14, 14]、[1, 224, 7, 7]
        result = self.pretrained(x,return_high_freq=True)  
        features = result[0] # result[0] 融合后的特征 result[1] 高频特征 result[2] 低频特征  
        depth = self.depth_head(features=features,size_h = x.shape[-2],size_w = x.shape[-1])
        return depth
    
    @torch.no_grad()
    def infer_image(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = torch.randn(1, 3, 224, 224).to(device)
        depth = self.forward(input_tensor) 
        
        return depth
    
    def load_pretrained(self, pretrained_path):  
        """加载预训练权重"""  
        checkpoint = torch.load(pretrained_path, map_location='cpu')  
        
        # 处理不同的权重格式  
        if 'model' in checkpoint:  
            state_dict = checkpoint['model']  
        else:  
            state_dict = checkpoint  
        
        # 加载权重到backbone  
        missing_keys, unexpected_keys = self.pretrained.load_state_dict(  
            state_dict, strict=False  
        )  
        
        print(f"Loaded pretrained weights from {pretrained_path}")  
        if missing_keys:  
            print(f"Missing keys: {missing_keys}")  
        if unexpected_keys:  
            print(f"Unexpected keys: {unexpected_keys}")
    