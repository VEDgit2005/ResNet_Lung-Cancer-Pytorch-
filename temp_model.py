import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a basic residual block for 1D data
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define the ResNet model for 1D gene expression data
class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_input_features, num_classes=2):
        super(ResNet1D, self).__init__()
        self.in_channels = 64 # Initial number of channels

        # Initial convolutional layer
        # Assuming input_features is the number of genes, and we treat it as the "width"
        # The input to Conv1d should be (batch_size, channels, length)
        # So, we'll treat the number of genes as 'length' and 1 as 'channels' initially,
        # then expand channels as we go deeper.
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Global Average Pooling and Fully Connected Layer
        # The output size of the last convolutional layer will depend on num_input_features
        # We'll use AdaptiveAvgPool1d to handle variable input sizes gracefully.
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes) # Output features from last layer

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride_val in strides:
            layers.append(block(self.in_channels, out_channels, stride_val))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input x is expected to be (batch_size, num_input_features)
        # We need to reshape it to (batch_size, 1, num_input_features) for Conv1d
        x = x.unsqueeze(1) # Adds a channel dimension

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1) # Flatten all dimensions except batch
        out = self.fc(out)
        return out

# Helper functions to create specific ResNet models
def ResNet18_1D(num_input_features, num_classes=2):
    return ResNet1D(BasicBlock1D, [2,2,2,2], num_input_features, num_classes)

def ResNet34_1D(num_input_features, num_classes=2):
    return ResNet1D(BasicBlock1D, [3,4,6,3], num_input_features, num_classes)

# Example usage:
if __name__ == '__main__':
    # Define the number of input features (genes)
    num_genes = 20000 # Example: 20,000 genes
    num_classes = 2 # Example: Lung Cancer (Positive/Negative)

    # Create a ResNet18_1D model
    model = ResNet18_1D(num_input_features=num_genes, num_classes=num_classes)
    print(model)

    # Create a dummy input tensor (batch_size, num_genes)
    # This simulates your gene expression data for one batch
    dummy_input = torch.randn(16, num_genes) # 16 samples, num_genes features

    # Pass the dummy input through the model
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be (batch_size, num_classes)
