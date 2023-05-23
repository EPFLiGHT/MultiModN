import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet(nn.Module):
    def __init__(self, *, state_size=0, freeze=False, pretrained_path=None, pretrained=True):
        super().__init__()

        if pretrained_path is not None and pretrained:
            raise ValueError(
                "Loading a pretrained ResNet should either be from torch.vision (pretrained=True) "
                "or from a checkpoint (pretrained_path) but not both."
            )

        # if pretrained, loads ResNet18 pretrained on ImageNet
        # self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.state_size = state_size

        self.fc = nn.Linear(512 + self.state_size, self.state_size)

        # load pre-trained ResNet from path
        if pretrained_path:
            model_dict = self.resnet.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {
                k: v for k, v in torch.load(pretrained_path).items() if k in model_dict
            }
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.resnet.load_state_dict(model_dict)

        # remove final classification layer
        self.resnet.fc = nn.Identity()

        if freeze:
            for p in self.resnet.parameters():
                p.requires_grad = False

    def forward(self, state, images):
        representations = self.resnet(images)
        output = self.fc(torch.cat([representations, state], dim=1))
        return output  # representations
