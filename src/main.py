import pytorch_lightning as pl
from pytorch_lightning.cli import ArgsType, LightningCLI

from frame_classifier.resnet_classifier import ResNetClassifier  # import my model
# from my_data_module import MyDataModule  # Replace with your actual data module

def cli_train(args: ArgsType = None):
    cli = LightningCLI(ResNetClassifier)
    result = cli.trainer.test(cli.model, datamodule=cli.datamodule)

if __name__ == "__main__":
    cli_train()  # don't provide any arguments so that cli is taking sys.args instead

    '''
    args = {
        "trainer": {
            "max_epochs": 3,
        },
        "model": {},
    }
    args["model"]["tune_fc_only"] = True

    cli_train(args)
    '''