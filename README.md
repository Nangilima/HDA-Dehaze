# HDA-Dehaze

## Haze-Density-Aware Non-Homogeneous Dehazing via Cascaded Mean-Teacher Diffusion Refinement

### Overview

HDA-Haze is a haze-density-aware framework for non-homogeneous image dehazing that synergistically combines cascaded mean-teacher learning with diffusion-based refinement. By explicitly modeling scene-dependent haze density and leveraging the generative capabilities of diffusion models, HDA-Haze enables robust training under extreme label scarcity while achieving state-of-the-art dehazing quality.

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Training

```bash
bash train.sh
```

#### Evaluation

```bash
python eval_diffusion.py --config_ddm configs.yml
```

### ToDo

- [ ] Implement diffusion model architecture
- [ ] Add distributed training support
- [ ] Release pre-trained models
- [ ] Update configuration files
- [ ] Add evaluation scripts with metrics

### Citation

If this project helps your research, please cite our paper:

