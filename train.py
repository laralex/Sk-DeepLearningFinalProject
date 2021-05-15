import sys
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
import torch
class CustomLightningCLI(LightningCLI):
    def __init__(self, config_path, *args, **kwargs):
        self.config_path = config_path
        super().__init__(*args, **kwargs)

    def parse_arguments(self):
        if self.config_path:
            self.config = self.parser.parse_path(self.config_path)
        else:
            self.config = self.parser.parse_args()

    def add_arguments_to_parser(self, parser):
        parser.add_argument('--experiment_name', default='DefaultModel')
        parser.add_argument('--checkpoint_monitor', default='loss')
        parser.add_argument('--checkpoint_monitor_mode', default='min')

    def before_fit(self):
        experiment_name = self.config['experiment_name']
        monitor         = self.config['checkpoint_monitor']
        monitor_mode    = self.config['checkpoint_monitor_mode']

        # Initialize TensorBoard logger
        logger = pl.loggers.TensorBoardLogger(save_dir=f'logs', name=experiment_name)
        self.trainer.logger = logger

        # Initialize saving of a best checkpoint
        callback = pl.callbacks.ModelCheckpoint(
            monitor=monitor,
            filename="{epoch:02d}-{" + monitor.replace('/', '_') + ":.3f}",
            mode=monitor_mode)
        self.trainer.callbacks.append(callback)


def main(config_path=None, gpus=1):
    cli = CustomLightningCLI(
        config_path,
        pl.LightningModule,
        pl.LightningDataModule,
        trainer_defaults={'gpus': gpus, 'benchmark': (gpus is not None)},
        seed_everything_default=42,
        subclass_mode_model=True,
        subclass_mode_data=True)

if __name__ == '__main__':
    main(gpus=1 if torch.cuda.is_available() else None)