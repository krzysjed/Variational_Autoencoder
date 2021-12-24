import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, figure
from torchvision.utils import make_grid
from Model import VAE
from torchvision.utils import save_image
import torch

if __name__ == "__main__":
    model = VAE().load_from_checkpoint(
        checkpoint_path=r"Path\file.ckpt")
    figure(figsize=(20, 8), dpi=300)

    rand = torch.randn(96, 64)  # number samples, size of bottleneck

    pred = model.decode(rand.to(model.device)).cpu()

    # save_image(pred, 'Images' + '.png')
    img = make_grid(pred).permute(1, 2, 0).numpy()
    plt.axis(False)
    imshow(img)
    plt.show()
