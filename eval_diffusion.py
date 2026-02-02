import argparse
import os
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from dataset import Data
from dataset import MyDataset
from models import DenoisingDiffusion, DiffusiveRestoration
import torch.distributed as dist
import torchvision
from torch.utils.data.distributed import DistributedSampler


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group(backend="nccl",init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(
    f"Global Rank: {dist.get_rank()}, "
    f"Local Rank: {rank}, "
    f"World Size: {dist.get_world_size()}"
    )

def load_config(config_arg, default_file):
    """Generic config loading function"""
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--{config_arg}", default=default_file, 
                       type=str, help=f"Path to {config_arg} config")
    args = parser.parse_args()
    
    try:
        with open(getattr(args, config_arg), "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(f"Error: Config file {getattr(args, config_arg)} not found")
    except yaml.YAMLError as exc:
        raise SystemExit(f"Config file parsing error: {exc}")

    return dict2namespace(config)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    print("labels ===", labels)
    return images, labels

def main(rank, world_size):
    setup(rank, world_size)
    config_ddm = load_config("config_ddm", "configs.yml")

    if torch.cuda.is_available():    
        print("GPU is available")  
    
    num_gpus = torch.cuda.device_count()  
    print("Number of GPUs Available: ", num_gpus)  
    
    for i in range(num_gpus):  
        print("GPU ", i, ": ", torch.cuda.get_device_name(i))  
    else:  
        print("No GPU available.")

    device = torch.device(f"cuda:{rank}")
    print("=> using device: {}".format(device))
    config_ddm.device = device
    
    torch.manual_seed(config_ddm.training.seed)
    np.random.seed(config_ddm.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config_ddm.training.seed)
    torch.backends.cudnn.benchmark = True

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    test_path = os.path.join(config_ddm.data.test_true_data_dir)
    test_dataset = MyDataset(test_path,
                    n=config_ddm.training.patch_n,
                    patch_size=config_ddm.data.image_load_size,
                    transforms=transforms,
                    parse_patches=False
                    )
    test_sampler = DistributedSampler(test_dataset, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config_ddm.sampling.batch_size, 
        sampler=test_sampler,
        num_workers=0, 
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        )

    print("=> creating diffusion model")
    diffusion = DenoisingDiffusion(config_ddm)
    model = DiffusiveRestoration(diffusion, config_ddm)

    model.restore(test_loader, r=config_ddm.data.grid_r)


if __name__ == '__main__':
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    torch.multiprocessing.set_start_method('spawn')
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        main,
        args=(world_size, ),
        nprocs=world_size,
        join=True
    )

