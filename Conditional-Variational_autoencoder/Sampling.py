import setuptools
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, figure
from torchvision.utils import make_grid
from Model import CVAE
from torchvision.utils import save_image
import torch
from DataLoader import DataModule


if __name__ == "__main__":

    model = CVAE().load_from_checkpoint(
      checkpoint_path=r"Path\file.ckpt")

    num = 96

    # Selecting attributes -> [0, 0, 1]
    y = torch.cat([torch.zeros(num, 1), torch.zeros(num, 1), torch.ones(num, 1)], dim=1)

    rand_v = torch.randn((num, 64))
    x_edit = torch.cat([rand_v, y], dim=1)

    pred = model.decode(x_edit.to(model.device)).cpu()

    #save_image(pred, 'Images' + '.png')
    img = make_grid(pred).permute(1, 2, 0).detach().numpy()

    figure(figsize=(20, 8), dpi=300)
    plt.axis(False)
    imshow(img)

    plt.show()
