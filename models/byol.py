import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import copy

def regression_loss(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2 - 2 * (x * y).sum(dim=-1)


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class BYOL(nn.Module):
    def __init__(self, backbone=resnet18()):
        super().__init__()

        self.backbone = backbone
        # projector
        self.projector = projection_MLP(backbone.output_dim)
        self.online_encoder = nn.Sequential(  # f encoder
            self.backbone,
            self.projector
        )
        self.m = 0.99
        self.online_predictor = prediction_MLP()

        self.target_encoder = copy.deepcopy(self.online_encoder)

        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x1, x2):
        online_proj_one = self.online_encoder(x1)
        online_proj_two = self.online_encoder(x2)

        p1 = self.online_predictor(online_proj_one)
        p2 = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_proj_one = self.target_encoder(x1)
            target_proj_two = self.target_encoder(x2)

        loss_one = regression_loss(p1, target_proj_two.detach())
        loss_two = regression_loss(p2, target_proj_one.detach())

        loss = loss_one + loss_two

        return {'loss': loss}
