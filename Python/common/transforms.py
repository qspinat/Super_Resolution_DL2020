import numpy as np
import torch
from torchvision import transforms
from skimage import transform
from common.constants import DEFAULT_INPUT_SIZE


class Resize(object):
  def __init__(self, output_size, scale_factor=2, same_size_input_label=False):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size
    self.scale_factor = scale_factor
    self.same_size_input_label = same_size_input_label

  def __call__(self, sample):
    image, label = sample['image'], sample['label']

    h, w = image.shape[:2]
    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    img_h = 2 * new_h if self.same_size_input_label else new_h
    img_w = 2 * new_w if self.same_size_input_label else new_w
    img = transform.resize(image, (img_h, img_w))
    lbl = transform.resize(label, (self.scale_factor * new_h, self.scale_factor * new_w))

    return {'image': img, 'label': lbl}


class RandomCrop(object):
  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size

  def __call__(self, sample):
    image, label = sample['image'], sample['label']

    h, w = image.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    image = image[top: top + new_h,
                  left: left + new_w]
    label = label[top: top + new_h,
                  left: left + new_w]

    return {'image': image, 'label': label}


class ToTensor(object):
  def __call__(self, sample):
    image, label = sample['image'], sample['label']

    image = image.transpose((2, 0, 1))
    label = label.transpose((2, 0, 1))
    
    return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label) }


def create_transforms(train_resize=DEFAULT_INPUT_SIZE, test_resize=DEFAULT_INPUT_SIZE, scale_factor=2, same_size_input_label=False):
  train_transforms = transforms.Compose([
      RandomCrop(scale_factor * train_resize),
      Resize(train_resize,scale_factor=scale_factor, same_size_input_label=same_size_input_label),
      ToTensor(),
  ])
  test_transforms = transforms.Compose([
      Resize(test_resize,scale_factor=scale_factor, same_size_input_label=same_size_input_label),
      ToTensor(),
  ])

  return train_transforms, test_transforms
