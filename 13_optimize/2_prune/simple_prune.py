import torch
import torch.nn.utils.prune

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square conv kernel
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120) # 5x5 image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        
    def forward(self, x):
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), (2, 2))
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.ReLU(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device=device)

print(list(model.conv1.named_parameters()))

torch.nn.utils.prune.random_unstructured(model.conv1, name='weight', amount=0.3)

print(list(model.conv1.named_parameters()))
