CUDA_VISIBLE_DEVICES=0 torchrun --master_port 6666 --nproc_per_node=1 tools/train.py configs/train_psm_cfg.py --launcher pytorch --validate --gpus 1
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 6666 --nproc_per_node=1 tools/test.py configs/test_psm_cfg.py --launcher pytorch --validate --gpus 1

conda create -n HODC python=3.9
conda activate HODC
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

find . -type d -name "__pycache__" -exec rm -rf {} + # clean __pycache__ recursively