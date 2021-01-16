import matplotlib.pyplot as plt
import torch
from torchvision import utils


class Visualizer():
  def __init__(self, figsize=(15,15)):
    self.figsize = figsize

  def subplot_torch_image(self, subplot: int, img: torch.Tensor, title: str = None):
    plt.subplot(subplot)
    
    grid = utils.make_grid(img)
    plt.imshow(grid.numpy().transpose((1,2,0)))
    
    if title is not None:
      plt.title(title)

  def visualize_sample(self, sample: torch.Tensor):
    image, label = sample['image'], sample['label']

    plt.figure(figsize=self.figsize)
    self.subplot_torch_image(121, image, 'Image')
    self.subplot_torch_image(122, label, 'Label')
