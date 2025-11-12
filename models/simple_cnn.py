# models/simple_cnn.py
import torch
import torch.nn as nn

__all__ = ["SimpleCNN"]


class SimpleCNN(nn.Module):
    """
    CIFAR-10 friendly CNN (expects 3×32×32).
    Blocks: (64)×2 → MaxPool → (128)×2 → MaxPool → (256)×2 → GAP → Linear(256→10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),  # 32→16
            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2),  # 16→8
            conv_block(128, 256),
            conv_block(256, 256),
            nn.AdaptiveAvgPool2d(1),  # → 256×1×1
        )

        self.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(256, num_classes))

        # Kaiming init for convs; Linear left to PyTorch default (reinit in GA stage if needed)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)  # (N, 256)
        return self.classifier(x)

    # ---- Helper the optimisers rely on ----
    def last_linear(self) -> nn.Linear:
        """Return the final fully connected layer (classifier head)."""
        return self.classifier[-1]
