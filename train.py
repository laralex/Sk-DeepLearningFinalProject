from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
class CustomLightningCLI(LightningCLI):
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


def main(gpus=1):
    cli = CustomLightningCLI(
        LightningModule,
        LightningDataModule,
        trainer_defaults={'gpus': gpus, 'benchmark': (gpus is not None)},
        seed_everything_default=42,
        subclass_mode_model=True,
        subclass_mode_data=True)

if __name__ == '__main__':
    main(gpus=None)