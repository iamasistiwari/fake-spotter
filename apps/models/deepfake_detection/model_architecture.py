import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class LightweightAttentionModule(nn.Module):
    """Simplified attention module with reduced dimensions"""
    def __init__(self, in_features):
        super(LightweightAttentionModule, self).__init__()
        self.query = nn.Linear(in_features, in_features // 16)  # Further reduced dimension
        self.key = nn.Linear(in_features, in_features // 16)
        self.value = nn.Linear(in_features, in_features)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Apply attention mechanism with reduced computation
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        scores = torch.bmm(query, key.transpose(1, 2))
        attention_weights = F.softmax(scores / (query.size(-1) ** 0.5), dim=2)
        
        out = torch.bmm(attention_weights, value)
        return self.gamma * out + x


class EfficientNetLiteFeatureExtractor(nn.Module):
    """Lightweight feature extractor based on EfficientNet-B0"""
    def __init__(self, img_size=224):  # Reduced image size
        super(EfficientNetLiteFeatureExtractor, self).__init__()
        # Use B0 instead of B3 - much smaller model
        self.efficientnet = models.efficientnet_b0(weights="IMAGENET1K_V1")
        
        # Remove the classification head
        self.efficientnet.classifier = nn.Identity()
        
        # B0 has only 1280 features vs 1536 in B3
        self.feature_dim = 1280
        
        # Freeze more layers to reduce trainable parameters
        for param in self.efficientnet.features[:5].parameters():
            param.requires_grad = False
        
        # Simplified frequency domain analysis with smaller kernel and fewer filters
        self.freq_conv = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3)  # Reduced from 64 filters and kernel size 15
        self.freq_bn = nn.BatchNorm2d(32)
        self.freq_pool = nn.AdaptiveAvgPool2d(1)
        self.freq_fc = nn.Linear(32, 128)  # Reduced from 256
        
        # Simpler fusion layer
        self.fusion = nn.Linear(self.feature_dim + 128, self.feature_dim // 2)  # Output half the features
    
    def forward(self, x):
        # Extract spatial features using EfficientNet
        spatial_features = self.efficientnet.features(x)
        spatial_features = self.efficientnet.avgpool(spatial_features)
        spatial_features = torch.flatten(spatial_features, 1)
        
        # Extract frequency domain features - simplified
        freq_features = self.freq_conv(x)
        freq_features = F.relu(self.freq_bn(freq_features))
        freq_features = self.freq_pool(freq_features)
        freq_features = torch.flatten(freq_features, 1)
        freq_features = F.relu(self.freq_fc(freq_features))
        
        # Fuse features
        combined_features = torch.cat([spatial_features, freq_features], dim=1)
        fused_features = F.relu(self.fusion(combined_features))
        
        return fused_features


class SimpleTemporalModule(nn.Module):
    """Simplified temporal inconsistency module"""
    def __init__(self, feature_dim):
        super(SimpleTemporalModule, self).__init__()
        self.conv_1d = nn.Conv1d(feature_dim, feature_dim//4, kernel_size=3, padding=1)  # Reduced output channels
        self.pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        x_reshaped = x.permute(0, 2, 1)
        # Always ensure float32 for this operation
        x_reshaped = x_reshaped.to(dtype=torch.float32)
        temp_features = F.relu(self.conv_1d(x_reshaped))
        temp_features = self.pool(temp_features).squeeze(-1)
        return temp_features


class FakeSpotterModel(nn.Module):
    """Lightweight CNN-LSTM architecture for deepfake detection"""
    def __init__(self, frame_count=10, img_size=224):  # Reduced frame count and image size
        super(FakeSpotterModel, self).__init__()
        self.frame_count = frame_count
        self.img_size = img_size
        
        # Lightweight feature extractor
        self.feature_extractor = EfficientNetLiteFeatureExtractor(img_size)
        
        # Get the reduced output feature dimension
        self.feature_dim = self.feature_extractor.feature_dim // 2  # Already halved in the extractor
        
        # Simplified LSTM - single layer, unidirectional
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=256,  # Reduced from 512
            num_layers=1,     # Reduced from 2
            batch_first=True,
            bidirectional=False,  # Changed to unidirectional
            dropout=0.3
        )
        
        # Simplified attention module
        self.attention = LightweightAttentionModule(256)  # Matches LSTM hidden size
        
        # Simplified temporal module
        self.temporal_module = SimpleTemporalModule(self.feature_dim)
        
        # Single dropout layer with moderate rate
        self.dropout = nn.Dropout(0.5)
        
        # Simplified classification layers
        self.fc1 = nn.Linear(256, 256)
        self.output = nn.Linear(256 + self.feature_dim//4, 1)  # Combined with temporal features
        
        # Single layer normalization
        self.layer_norm = nn.LayerNorm(256)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Process frames in chunks to save memory
        chunk_size = 5  # Process 5 frames at a time to save memory
        frame_features_chunks = []

        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk = x[:, i:end_idx, :, :, :]
            chunk_flat = chunk.reshape(-1, c, h, w)
            
            # Use mixed precision only for feature extraction
            with torch.amp.autocast(device_type=torch.device('cuda' if torch.cuda.is_available() else "cpu").type, enabled=True):
                chunk_features = self.feature_extractor(chunk_flat)
            
            # Ensure correct dtype for LSTM
            chunk_features = chunk_features.to(dtype=torch.float32)
            chunk_features = chunk_features.view(batch_size, end_idx-i, -1)
            frame_features_chunks.append(chunk_features)
            
        # Concatenate all chunks
        frame_features = torch.cat(frame_features_chunks, dim=1)
        
        # Extract temporal features
        temp_features = self.temporal_module(frame_features)
        
        # Make sure we're operating in float32 for the LSTM
        frame_features = frame_features.to(dtype=torch.float32)
        
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(frame_features)
        
        # Apply attention
        attended_features = self.attention(lstm_out)
        
        # Pool across time dimension
        sequence_features = torch.mean(attended_features, dim=1)
        
        # Apply normalization and dropout
        sequence_features = self.layer_norm(sequence_features)
        sequence_features = self.dropout(sequence_features)
        
        # Dense layers
        dense = self.relu(self.fc1(sequence_features))
        
        # Combine with temporal features
        combined = torch.cat([dense, temp_features], dim=1)
        combined = self.dropout(combined)
        
        # Output probability
        output = self.sigmoid(self.output(combined))
        
        return output


class FocalLoss(nn.Module):
    """Focal Loss for dealing with class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)