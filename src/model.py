"""
Model architecture: ResNet-50 with emotion classification head.
Supports transfer learning with staged fine-tuning.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import config


class EmotionDetector(nn.Module):
    """
    ResNet-50 based emotion detector.
    Replaces the final classification layer for emotion classes.
    """
    
    def __init__(self, num_classes=config.NUM_EMOTIONS, pretrained=True, freeze_backbone=True):
        """
        Args:
            num_classes (int): Number of emotion classes
            pretrained (bool): Load pretrained ImageNet weights
            freeze_backbone (bool): Freeze backbone layers for transfer learning
        """
        super(EmotionDetector, self).__init__()
        
        # Load pretrained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get number of input features for the classifier
        num_features = self.backbone.fc.in_features
        
        # Replace classification head
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Store reference to original layers for staged fine-tuning
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
        self.fc = self.backbone.fc
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        """Freeze all backbone layers except the classification head."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_early_layers(self, num_layers=2):
        """
        Freeze early layers (layer1 and layer2) for discriminative fine-tuning.
        
        Args:
            num_layers (int): Number of early ResNet blocks to freeze (1-4)
        """
        layers_to_freeze = []
        if num_layers >= 1:
            layers_to_freeze.append(self.layer1)
        if num_layers >= 2:
            layers_to_freeze.append(self.layer2)
        if num_layers >= 3:
            layers_to_freeze.append(self.layer3)
        if num_layers >= 4:
            layers_to_freeze.append(self.layer4)
        
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input images of shape (B, 3, 224, 224)
        
        Returns:
            torch.Tensor: Emotion logits of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def get_discriminative_lr_groups(self, base_lr=0.0001):
        """
        Get parameter groups with discriminative learning rates.
        Early layers get lower LR, later layers get higher LR.
        
        Args:
            base_lr (float): Base learning rate for early layers
        
        Returns:
            list: List of parameter groups for optimizer
        """
        param_groups = [
            {'params': self.layer1.parameters(), 'lr': base_lr * 0.1},
            {'params': self.layer2.parameters(), 'lr': base_lr * 0.5},
            {'params': self.layer3.parameters(), 'lr': base_lr},
            {'params': self.layer4.parameters(), 'lr': base_lr * 2},
            {'params': self.fc.parameters(), 'lr': base_lr * 10},
        ]
        return param_groups


def create_model(num_classes=config.NUM_CLASSES, pretrained=True, freeze_backbone=True):
    """
    Factory function to create emotion detector model.
    
    Args:
        num_classes (int): Number of emotion classes
        pretrained (bool): Load pretrained weights
        freeze_backbone (bool): Freeze backbone layers
    
    Returns:
        EmotionDetector: Model instance
    """
    model = EmotionDetector(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    # Move to device
    model.to(config.DEVICE)
    
    return model


def count_parameters(model):
    """Count total trainable and non-trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


if __name__ == '__main__':
    # Test model creation and parameter counting
    print("Creating EmotionDetector model...")
    model = create_model(freeze_backbone=True)
    
    params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224).to(config.DEVICE)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")
