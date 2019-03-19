import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


CONFIG = [64, 'M', 128, 'M', 256, 256, 'M'] 

def get_model(model_path=None, cfg=CONFIG, in_channels=3, gpu=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    layers += [
        Flatten(),
        nn.Linear(16384, 1024),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(1024, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 200),
    ]
    model = nn.Sequential(*layers)
    if model_path is not None:
        if gpu:
            model.load_state_dict(torch.load(model_path, map_location="cuda")) 
            device = torch.device("cuda")
            model.to(device)
        else:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))  
            device = torch.device("cpu")
            model.to(device)
    # довольно важно также дампить параметры оптимизатора, но в этой работе опустим
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    return model, opt, loss_fn
