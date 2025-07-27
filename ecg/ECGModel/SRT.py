import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchmetrics
import lightning as L

class ScanningModule(nn.Module):
    def __init__(self):
        super(ScanningModule, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 2, kernel_size=17, padding="same"),
                nn.ReLU(),
                nn.BatchNorm1d(2),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(2, 4, kernel_size=11, padding="same"),
                nn.ReLU(),
                nn.BatchNorm1d(4),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ) for _ in range(12)  
        ])

    def forward(self, x):
        features = []
        for i, channel in enumerate(self.branches):
            lead = x[:, i, :].unsqueeze(1) 
            feat = channel(lead).squeeze(-1) 
            features.append(feat)
        
        x = torch.cat(features, dim=1) 
        return x

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, groups):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm1d(growth_rate)
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels=in_channels, 
                              out_channels=growth_rate, 
                              kernel_size=3, stride=1, padding="same", 
                              groups=groups)
    
    def forward(self, x):   
        out = self.bn(self.relu(self.conv(x)))
        return torch.cat([x, out], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, groups):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i*growth_rate, growth_rate, groups))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class ReadingModule(nn.Module):
    def __init__(self):
        super(ReadingModule, self).__init__()
        # Dense Block1
        self.db1 = DenseBlock(in_channels=48, growth_rate=12, num_layers=2, groups=12)
        self.trans1 = TransitionLayer(72, 36)  # 48 + 2*12 = 72
        # Dense Block2
        self.db2 = DenseBlock(in_channels=36, growth_rate=12, num_layers=7, groups=12)
        self.trans2 = TransitionLayer(120, 60)  # 36 + 7*12 = 120
        # Dense Block3
        self.db3 = DenseBlock(in_channels=60, growth_rate=12, num_layers=2, groups=12)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  
    
    def forward(self, x):
        x = self.db1(x)  
        x = self.trans1(x)  
        x = self.db2(x)  
        x = self.trans2(x) 
        x = self.db3(x)  
        x = self.pool3(x) 
        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerEncoderModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # Input: (batch, seq_len, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class ThinkingModule(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward):
        super(ThinkingModule, self).__init__()
        self.transformer_encoder = TransformerEncoderModel(d_model=d_model, nhead=nhead,
                num_layers=num_encoder_layers, dim_feedforward=dim_feedforward)
        self.dropout = nn.Dropout(0.5)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):   
        # x shape: (batch, 84, 62) -> transpose to (batch, 62, 84)
        x = x.transpose(1, 2)  # (batch, seq_len=62, d_model=84)
        x = self.transformer_encoder(x)
        x = self.gap(x)
        x = self.dropout(x)
        return x

class SRTNet(nn.Module):
    def __init__(self, num_classes=2):  
        super(SRTNet, self).__init__()
        self.scanning = ScanningModule()
        self.reading = ReadingModule()
        self.thinking = ThinkingModule(
            d_model=84,
            nhead=3, 
            num_encoder_layers=2,
            dim_feedforward=200)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(62, num_classes)
        )
        
    def forward(self, x):
        x = self.scanning(x)  
        x = self.reading(x)   
        x = self.thinking(x)
        x = self.classifier(x) 
        return x

class SRT(L.LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = SRTNet(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)   
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-5)
        return optimizer
