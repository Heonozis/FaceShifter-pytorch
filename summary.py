from torchsummary import summary
from network.aei import AEI_Net
import torch
device = torch.device("cpu")
net = AEI_Net().to(device)
summary(net, input_size=[(3, 64, 64), (512,)])

