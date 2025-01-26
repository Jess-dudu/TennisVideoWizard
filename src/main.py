import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from frame_classifier.resnet_classifier import ResNetClassifier  # import my model
# from my_data_module import MyDataModule  # Replace with your actual data module

def main():
    cli = LightningCLI(ResNetClassifier)

if __name__ == "__main__":
    main()