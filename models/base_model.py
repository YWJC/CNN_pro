import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MultiTaskModel(nn.Module):
    """多任务学习模型：同时进行分类和回归"""
    def __init__(self, backbone_name='resnet18', num_classes=3, pretrained=True):
        super(MultiTaskModel, self).__init__()
        
        # 创建骨干网络
        self.backbone_name = backbone_name
        self.feature_extractor = self._create_backbone(backbone_name, pretrained)
        
        # 根据不同骨干网络获取特征维度
        if backbone_name.startswith('resnet'):
            feature_dim = 512
        elif backbone_name.startswith('efficientnet'):
            feature_dim = 1280
        elif backbone_name.startswith('mobilenet'):
            feature_dim = 1280
        elif backbone_name.startswith('vgg'):
            feature_dim = 4096  # VGG特征维度
        else:
            feature_dim = 512
        
        # 分类分支
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 回归分支
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 将输出限制在0-1之间
        )
    
    def _create_backbone(self, backbone_name, pretrained):
        """创建不同的骨干网络"""
        if backbone_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            # 移除最后的全连接层
            modules = list(model.children())[:-1]
            return nn.Sequential(*modules)
        elif backbone_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            modules = list(model.children())[:-1]
            return nn.Sequential(*modules)
        elif backbone_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            # 添加自适应池化层以确保输出特征维度为1280
            return nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )
        elif backbone_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
            # 添加自适应池化层以确保输出特征维度为1280
            return nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )
        elif backbone_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
            # 获取特征提取部分
            return nn.Sequential(*list(model.features) + [nn.AdaptiveAvgPool2d((7, 7))])
        elif backbone_name == 'vgg19':
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
            # 获取特征提取部分
            return nn.Sequential(*list(model.features) + [nn.AdaptiveAvgPool2d((7, 7))])
        else:
            raise ValueError(f"不支持的骨干网络: {backbone_name}")
    
    def forward(self, x):
        """前向传播"""
        # 提取特征
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        
        # 如果是VGG，需要额外的全连接特征层
        if self.backbone_name.startswith('vgg'):
            features = F.relu(nn.Linear(25088, 4096).to(features.device)(features))
            features = F.dropout(features, p=0.5, training=self.training)
            features = F.relu(nn.Linear(4096, 4096).to(features.device)(features))
            features = F.dropout(features, p=0.5, training=self.training)
        
        # 分类输出
        class_output = self.classifier(features)
        
        # 回归输出
        reg_output = self.regressor(features)
        
        return class_output, reg_output
    
    def predict(self, x):
        """推理模式：先定性再定量"""
        self.eval()
        with torch.no_grad():
            class_output, reg_output = self.forward(x)
            # 获取分类预测
            class_pred = torch.argmax(class_output, dim=1)
            # 将回归输出还原为浓度值(0-1000nM)
            concentration_pred = reg_output.squeeze() * 1000
            
        self.train()
        return class_pred, concentration_pred