import os
from pathlib import Path
import yaml
import torch

def find_dataset_subdir(params_dict, datasets_root):
    for root, subdirs, _ in os.walk(datasets_root, topdown=False):
        for version in subdirs:
            root = Path(root)
            candidate_yaml = root/version/'signal_hparams.yaml'
            if os.path.exists(candidate_yaml):
                with open(candidate_yaml, 'r') as stream:
                    candidate_hparams = yaml.safe_load(stream)
                    if candidate_hparams == params_dict:
                        print('Matched dataset:', candidate_yaml)
                        return root/version
                    else:
                        print("Not matched dataset:", candidate_yaml, [f"ck={k} cv={v} dv={params_dict[k]}" for k, v in candidate_hparams.items() if params_dict[k] != v])
    return None

def load_from_subdir(path, data_type):
    assert os.path.exists(path/'signal_hparams.yaml')
    return concat_files(path/'train', data_type), concat_files(path/'val', data_type), concat_files(path/'test', data_type)

def concat_files(root_path, data_type):
    assert os.path.exists(root_path)
    files = [os.path.join(root_path, f) for f in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, f))]
    tensors = [torch.load(file).type(data_type) for file in files]
    if len(tensors) > 1:
        return torch.cat(tensors, dim=1)
    elif len(tensors) == 1:
        return tensors[0]
    else:
        return None