# HDA-Dehaze

## Haze-Density-Aware Single Image Dehazing via Progressive Mean-Teacher Diffusion Learning

### Overview

HDA-Dehaze is a single image dehazing method based on progressive mean-teacher diffusion learning. This approach achieves high-quality dehazing results through density-aware haze modeling combined with the powerful generative capabilities of diffusion models.

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

