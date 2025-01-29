import warnings
# warnings.filterwarnings("ignore")
import pytorch_lightning as pl
from pytorch_lightning.cli import ArgsType, LightningCLI

from resnet_classifier import ResNetClassifier

def cli_main(args: ArgsType = None):
    cli = LightningCLI(ResNetClassifier, args=args)
    # cli.trainer.fit(cli.model)
    result = cli.trainer.test(cli.model, datamodule=cli.datamodule)

if __name__ == "__main__":
    cli_main()

    '''
    # # Instantiate Model
    save_path = args.save_path if args.save_path is not None else "./models"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename="resnet-model-{epoch}-{val_loss:.2f}-{val_acc:0.2f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min")

    # Instantiate lightning trainer and train model
    trainer_args = {
        "accelerator": "gpu" if args.gpus else "cpu",
        "devices": args.gpus if args.gpus > 0 else "auto",
        "strategy": "ddp" if args.gpus > 1 else "auto",
        "max_epochs": args.num_epochs,
        "callbacks": [checkpoint_callback],
        "precision": 16 if args.mixed_precision else 32,
        "default_root_dir": save_path,
    }
    print(trainer_args)
    trainer = pl.Trainer(**trainer_args)

    trainer.fit(model)

    if args.test_set:
        trainer.test(model)
    # Save trained model weights
    torch.save(trainer.model.resnet_model.state_dict(), save_path + "/trained_model.pt")
    '''