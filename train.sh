export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0

mkdir -p ./logs

accelerate launch --mixed_precision fp16 --num_processes 1 train_diffusion.py 2>&1 | tee -a ./logs/training_$(date +%Y%m%d_%H%M%S).log
