import torch
import torch.nn as nn
from advanced_3dcnn import Advanced3DCNN  # Import your 3D CNN model
from model_embedder import HybridEmbed  # Import your ConvNeXt-Swin model

class HybridModel(nn.Module):
    def __init__(self, num_classes, cnn_model, swin_model):
        super(HybridModel, self).__init__()

        # 3D CNN model
        self.cnn_model = cnn_model

        # ConvNeXt-Swin model
        self.swin_model = swin_model

        # Additional layers for combining features
        self.fc1 = nn.Linear(
            cnn_model.fc.in_features + swin_model.fc.in_features, 512
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x_cnn, x_swin):
        # Forward pass for the 3D CNN model
        cnn_features = self.cnn_model(x_cnn)

        # Forward pass for the ConvNeXt-Swin model
        swin_features = self.swin_model(x_swin)

        # Combine features
        combined_features = torch.cat((cnn_features, swin_features), dim=1)

        # Additional layers for feature combination
        combined_features = self.fc1(combined_features)
        combined_features = self.relu(combined_features)
        combined_features = self.dropout(combined_features)
        output = self.fc2(combined_features)

        return output

# Example usage:

# Initialize 3D CNN model and ConvNeXt-Swin model
num_classes = 10  # Adjust based on your classification task
cnn_model = Advanced3DCNN(num_classes)
swin_model = HybridEmbed(num_classes)

# Initialize the hybrid model
hybrid_model = HybridModel(num_classes, cnn_model, swin_model)

# Print the hybrid model architecture
print(hybrid_model)

