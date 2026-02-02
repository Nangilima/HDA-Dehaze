import argparse
import os
import sys
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torch.distributed as dist
import argparse
import time
import logging
from pathlib import Path

from dataset import Data
from models import DenoisingDiffusion

def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_{int(time.time())}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    return logging.getLogger(__name__)

def validate_config(config):
    required_sections = ['training', 'data', 'diffusion']
    for section in required_sections:
        if not hasattr(config, section):
            raise ValueError(f"Missing required section in config: {section}")
    
    training = config.training
    if training.batch_size <= 0:
        raise ValueError("Batch size must be greater than 0")
    
    if not os.path.exists(config.data.train_dataset_dir):
        raise ValueError(f"Training dataset path does not exist: {config.data.train_dataset_dir}")
    
    return True
def load_config(config_arg, default_file):
    parser = argparse.ArgumentParser(description='Non-uniform haze removal training')
    parser.add_argument(f"--{config_arg}", default=default_file, 
                       type=str, help=f"Path to {config_arg} config")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, default="", 
                       help="Checkpoint file path")
    parser.add_argument("--log_dir", type=str, default="./logs", 
                       help="Directory to save logs")
    args = parser.parse_args()
    
    try:
        with open(getattr(args, config_arg), "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(f"Error: Config file {getattr(args, config_arg)} not found")
    except yaml.YAMLError as exc:
        raise SystemExit(f"Config file parsing error: {exc}")

    config_ns = dict2namespace(config)
    
    config_ns.args = args
    
    return config_ns

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    config_ddm = load_config("config_ddm", "configs.yml")
    
    logger = setup_logging(config_ddm.args.log_dir)
    logger.info("=" * 80)
    logger.info("Starting non-uniform haze removal training")
    logger.info("=" * 80)
    
    try:
        validate_config(config_ddm)
        logger.info("Config validation passed")
    except ValueError as e:
        logger.error(f"Config validation failed: {e}")
        return
    
    torch.manual_seed(config_ddm.training.seed)
    np.random.seed(config_ddm.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config_ddm.training.seed)
        logger.info(f"Training with GPU, number of devices: {torch.cuda.device_count()}")
    else:
        logger.warning("No GPU detected, will use CPU for training")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    logger.info("Training configuration:")
    logger.info(f"  - Batch size: {config_ddm.training.batch_size}")
    logger.info(f"  - Learning rate: {getattr(config_ddm.training, 'learning_rate', '0.00005')}")
    logger.info(f"  - Pre-model training iterations: {config_ddm.training.n_iters_pre}")
    logger.info(f"  - Random seed: {config_ddm.training.seed}")
    
    try:
        logger.info("Creating diffusion model...")
        diffusion = DenoisingDiffusion(config_ddm)
        
        if config_ddm.args.resume and config_ddm.args.checkpoint:
            if os.path.exists(config_ddm.args.checkpoint):
                logger.info(f"Resuming training from checkpoint: {config_ddm.args.checkpoint}")
                diffusion.load_checkpoint(config_ddm.args.checkpoint)
            else:
                logger.warning(f"Checkpoint file not found: {config_ddm.args.checkpoint}")
        
        logger.info("Starting training process...")
        start_time = time.time()
        
        diffusion.train()
        
        total_time = time.time() - start_time
        logger.info(f"Training completed! Total time: {total_time:.2f} seconds")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.exception("Detailed error information:")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Training finished, resources cleaned up")


if __name__ == "__main__":
    main()
