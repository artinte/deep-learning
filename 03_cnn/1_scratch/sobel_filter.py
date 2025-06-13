import torch
from PIL import Image
import torch
import torchvision
from matplotlib import pyplot

img = Image.open('docs/res/03/lena.jpg').convert('L')
img_tensor = torchvision.transforms.ToTensor()(img).unsqueeze(0)

sobel_y = torch.tensor([
    [[[-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]]]
], dtype=torch.float32)

sobel_x = torch.tensor([
    [[[-1, -2, -1],
      [0,  0,  0],
      [1,  2,  1]]]
], dtype=torch.float32)

# output demension: [B, C, H, W]
grad_x = torch.nn.functional.conv2d(img_tensor, sobel_y, padding=1)
grad_y = torch.nn.functional.conv2d(img_tensor, sobel_x, padding=1)

pyplot.imshow(grad_x[0, 0].detach(), cmap='gray')
pyplot.imsave('temp/sobel_y.png', grad_x[0, 0].detach().numpy(), cmap='gray')
pyplot.show()

pyplot.imshow(grad_y[0, 0].detach(), cmap='gray')
pyplot.imsave('temp/sobel_x.png', grad_y[0, 0].detach().numpy(), cmap='gray')
pyplot.show()
