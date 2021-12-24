import torch
from DataLoader import DataModule
from Model import VAE
from pytorch_lightning import Trainer


if __name__ == "__main__":

    data = DataModule()
    model = VAE().load_from_checkpoint(
        checkpoint_path=r"Path\file.ckpt")

    with torch.no_grad():
        trainer = Trainer(log_every_n_steps=1, gpus=0, limit_val_batches=0.0,
                          checkpoint_callback=False,
                          logger=False)
        trainer.test(model, data)
