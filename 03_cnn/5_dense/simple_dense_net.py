import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Bottleneck(nn.Module):
    """Bottleneck layer for DenseNet (1x1 conv -> 3x3 conv)"""
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        # 1x1 convolution to reduce number of channels (bottleneck)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        
        # 3x3 convolution to produce new features
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # Concatenate input with output (dense connection)
        out = torch.cat([x, out], 1)
        return out

class Transition(nn.Module):
    """Transition layer between dense blocks (reduces channels and spatial dimensions)"""
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

class DenseNet(nn.Module):
    """DenseNet main architecture"""
    def __init__(self, block, num_blocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        
        # Initial convolution
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        
        # Dense blocks and transition layers
        self.dense1 = self._make_dense_block(block, num_blocks[0], num_channels)
        num_channels += num_blocks[0] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans1 = Transition(num_channels, out_channels)
        num_channels = out_channels
        
        self.dense2 = self._make_dense_block(block, num_blocks[1], num_channels)
        num_channels += num_blocks[1] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans2 = Transition(num_channels, out_channels)
        num_channels = out_channels
        
        self.dense3 = self._make_dense_block(block, num_blocks[2], num_channels)
        num_channels += num_blocks[2] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans3 = Transition(num_channels, out_channels)
        num_channels = out_channels
        
        self.dense4 = self._make_dense_block(block, num_blocks[3], num_channels)
        num_channels += num_blocks[3] * growth_rate
        
        # Final layers
        self.bn = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_classes)

    def _make_dense_block(self, block, num_layers, in_channels):
        """Create a dense block with specified number of layers"""
        layers = []
        for _ in range(num_layers):
            layers.append(block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def densenet121():
    """DenseNet-121 configuration"""
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)

def densenet169():
    """DenseNet-169 configuration"""
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32)

def densenet201():
    """DenseNet-201 configuration"""
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32)

def densenet161():
    """DenseNet-161 configuration"""
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48)

# Example usage
if __name__ == "__main__":
    # Create DenseNet-121 model
    model = densenet121()
    
    # Test with random input (CIFAR-10 size: 3 channels, 32x32)
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print model summary (requires torchsummary)
    summary(model, (3, 32, 32))
