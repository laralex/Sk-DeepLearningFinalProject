# Signal recovery from nonlinear distortion in optical communications
The final project of Deep Learning 2021 course at Skoltech, Russia.

## ❗ Attention ❗ We kindly ask the evaluating committee to consider the [`PRESENTATION.pdf`](PRESENTATION.pdf) and [`REPORT.pdf`](REPORT.pdf) from there, rather than Canvas. They were significantly improved the night after the deadline and is very different from the version sent to the Canvas. 
Team members:
* Ilya Kuk
* Razan Dibo
* Mohammed Deifallah
* Sergei Gostilovich
* Alexey Larionov
* Stanislav Krikunov
* Alexander Blagodarnyi

Inspired by the papers:

1. [Advancing theoretical understanding and practical performance of signal processing for nonlinear optical communications through machine learning](https://www.nature.com/articles/s41467-020-17516-7).
2. [Fundamentals of Coherent Optical Fiber Communications ](https://www.osapublishing.org/jlt/abstract.cfm?URI=jlt-34-1-157)

### Brief repository overview
* [`train.py`](train.py) - entry point for training of models (see
  **Reproduce training and inference** section)
* [`notebooks/training.ipynb`](notebooks/training.ipynb) - a quickstart Jupyter
  notebook for training or loading from a checkpoint
* [`configs/`](configs/) - YAML files that define each experiment's parameters
* [`data/`](data/) - definitions of datasets (either preloaded or generated)
* [`materials/`](materials/) - supplementary materials like reports, plots
* [`models/`](models/) - definitions of models and their training process (optimizers, learning rate schedulers)
* [`auxiliary/`](auxiliary/) - supporting files with utility functions
* 👉 [`PRESENTATION.pdf`](PRESENTATION.pdf) - final presentation
* 👉 [`REPORT.pdf`](REPORT.pdf) - project final report
* 👉 [`VIDEO_PRESENTATION.txt`](VIDEO_PRESENTATION.txt) - link to video with project presentation
### Requirements
A GPU is recommended to perform the experiments. You can use [Google
Colab](colab.research.google.com) with Jupyter notebooks provided in
[`notebooks/`](notebooks/) folder

Main prerequisites are:

- [`pytorch`](http://pytorch.org/) for models training
- [`pytorch-lightning`](https://www.pytorchlightning.ai/) with [`LightningCLI`](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html#lightningcli) for CLI tools and low-boilerplate models training
- [`numpy`](https://anaconda.org/anaconda/numpy) for data generation

Optional:
- [`google-colab`](https://anaconda.org/conda-forge/google-colab) - if you want to mount a Google Drive to your Jupyter Notebook to store training artifacts
- [`gdown`](https://anaconda.org/conda-forge/gdown) - if you want to download checkpoints from Google Drive by ID
- [`tensorboard`](https://anaconda.org/conda-forge/tensorboard) - if you want to view training/validation metrics of different experiments
- [`torchvision`](https://anaconda.org/pytorch/torchvision) - only if you want to debug the workflow with a trivial MNIST classifier example


<!TODO:
For convenience, a ready to use conda [environment](environment.yml) is provided. 
To create a new python environment with all the required packages, you can run:
```shell
conda env create -f environment.yml
conda activate dyconv
```
!>

### Reproduce training and inference
The easiest way to start training of one of experiments listed in [`configs/`](configs/), is to run
```bash
python train.py --config configs/your_chosen_experiment.yaml
```
After that you'll find new folders `downloads/` with external downloaded files (like datasets) and `logs/` which will contain folders for each distinct experiment. Under each such experiment folder you'll find results of all the runs of this very same experiment, namely folders like `version_0/`, `version_1/`, etc, which would finally contain:
- `config.yaml` with the parameters of the experiment for reproducibility (same parameters as in `your_chosen_experiment.yaml`)
- `events.out.tfevents...` file with logs of TensorBoard, ready to be visualized in it
- `checkpoints/` directory with the best epoch checkpoint and the lastest epoch ckeckpoint (you can use those to resume training from them, or load them for inference)

A better approach to start training (or resuming or loading for inference), would be to use [`notebooks/training.ipynb`](notebooks/training.ipynb) Jupyter Notebook. In the first section you can set the parameters of further work. Other sections don't usually need any adjustments. After you "Run All" the notebook, either a training will start (a new one, or resumed), or only the model weights will be loaded (if you've chosen to `'load_model'`, see the notebook). 

Anyway after the notebook has been run completely, you should be given `model` variable of type [`pytorch_lightning.LightningModule`](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#inference). You can do inference with it suing `model.forward(x)`.
