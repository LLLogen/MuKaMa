# MuKaMa：Multi-Knowledge-aware Matcher
# Requirements

* Python 3.7.7
* PyTorch 1.9
* HuggingFace Transformers 4.9.2

Install required packages
```
conda install -c conda-forge nvidia-apex
pip install -r requirements.txt
```
## Training with MuKaMa

To train the matching model with MuKaMa:
```
CUDA_VISIBLE_DEVICES=0 python train_MuKaMa.py \
  --task Structured/Beer \
  --batch_size 32 \
  --max_len 128 \
  --lr 3e-5 \

The meaning of the flags:
* ``--task``: the name of the tasks (see ``configs.json``)
* ``--batch_size``, ``--max_len``, ``--lr``, ``--n_epochs``: the batch size, max sequence length, learning rate, and the number of epochs
* ``--save_model``: if this flag is on, then save the checkpoint to ``{logdir}/{task}/model.pt``.
