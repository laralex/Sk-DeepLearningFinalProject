import gc
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI, SaveConfigCallback

class CustomLightningCLI(LightningCLI):
    def __init__(self, root_dir, config_path, *args, do_fit=True, version=None, dataset_root_path=None, model_override=None, data_override=None, **kwargs):
        self.root_dir = root_dir
        self.config_path = config_path
        self.do_fit = do_fit
        self.version = version
        self.dataset_root_path = dataset_root_path
        self.model_override = model_override
        self.data_override = data_override
        super().__init__(*args, **kwargs)

    def parse_arguments(self):
        if self.config_path:
            self.config = self.parser.parse_path(self.config_path)
        else:
            self.config = self.parser.parse_args()
        if 'load_dataset_root_path' in self.config['data']['init_args']:
            self.config['data']['init_args']['load_dataset_root_path'] = self.dataset_root_path

    def add_arguments_to_parser(self, parser):
        parser.add_argument('--experiment_name', default='DefaultModel')
        parser.add_argument('--checkpoint_monitor', default='loss')
        parser.add_argument('--checkpoint_monitor_mode', default='min')

    def before_fit(self):
        experiment_name = self.config['experiment_name']
        monitor         = self.config['checkpoint_monitor']
        monitor_mode    = self.config['checkpoint_monitor_mode']

        # Initialize TensorBoard logger
        logger = pl.loggers.TensorBoardLogger(
            save_dir=f'{self.root_dir}/logs',
            name=experiment_name,
            version=self.version)
        self.trainer.logger = logger

        # Initialize saving of a best checkpoint
        callback = pl.callbacks.ModelCheckpoint(
            monitor=monitor,
            filename="{epoch:02d}-{" + monitor + ":.3f}",
            mode=monitor_mode,
            )
        self.trainer.callbacks.append(callback)

    def fit(self):
        print(f'Loading the dataset from {self.dataset_root_path}')
        if self.do_fit:
            super().fit()

    def instantiate_datamodule(self) -> None:
        if self.data_override is None:
            super().instantiate_datamodule()
        else:
            self.datamodule = self.data_override

    def instantiate_model(self) -> None:
        if self.model_override is None:
            super().instantiate_model()
        else:
            self.model = self.model_override

def main(root_dir='.', config_path=None, gpu_indices=1, checkpoint_kwargs=None, dataset_root_path=None, model_override=None, data_override=None):
    """
    root_dir: absolute path to where to put logs/ directory
    config_path: absolute path to a file with configuration in YAML format
    gpus: indices of GPUs to use, None for CPU, list for multiple GPUs
    checkpoint: a tuple of (source_str, type, action), where source_str is a
    link/path to a checkpoint, type='link'|'path', and action='resume_training'|'load_model'
    """
    do_fit = True
    trainer_defaults = {'gpus': gpu_indices, 'benchmark': (gpu_indices is not None)}
    version = None
    save_config_callback = SaveConfigCallback

    # load checkpoint
    if checkpoint_kwargs is not None:
        save_config_callback = None
        trainer_defaults['resume_from_checkpoint'] = checkpoint_kwargs['source']
        if checkpoint_kwargs.get('action', 'load_model') == 'load_model':
            do_fit = False
        version = checkpoint_kwargs.get('version', 'from_checkpoint')
    try:
        torch.multiprocessing.set_start_method('spawn')
    except: pass
    old_cwd = os.getcwd()
    # os.chdir(root_dir)
    try:
        cli = CustomLightningCLI(
            root_dir,
            config_path,
            pl.LightningModule,
            pl.LightningDataModule,
            dataset_root_path=dataset_root_path,
            model_override=model_override,
            data_override=data_override,
            do_fit=do_fit,
            version=version,
            trainer_defaults=trainer_defaults,
            save_config_callback=save_config_callback,
            seed_everything_default=42,
            subclass_mode_model=True,
            subclass_mode_data=True)
    finally:
        # os.chdir(old_cwd)
        gc.collect()

    return cli.model, cli.datamodule

if __name__ == '__main__':
    main(gpu_indices=1 if torch.cuda.is_available() else None)