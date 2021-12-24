from pytorch_lightning import  Trainer
from DataLoader import DataModule
from Model import VAE
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":

    num_batch = 64
    num_workers = 3
    data = DataModule(num_batch, num_workers)

    model = VAE(1, 2000)

    # Version with W&B and gpu acceleration
    # wandb_logger = WandbLogger(project="VAE", name="VAE_ver1", log_model=False)
    # trainer = Trainer(log_every_n_steps=1, gpus=2,accelerator='ddp', limit_val_batches=0.0, max_epochs=16000,
    # checkpoint_callback=True, progress_bar_refresh_rate=20, logger=wandb_logger)

    # Basic version running on the processor
    trainer = Trainer(log_every_n_steps=1, gpus=0, limit_val_batches=0.0, max_epochs=100, checkpoint_callback=False,
                      logger=False)

    trainer.fit(model.train(), data)
