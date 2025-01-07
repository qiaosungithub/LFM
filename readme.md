# Flow Matching in Latent Space with SQA documentation

## Env

```
conda create -n LFM python=3.10
conda activate LFM
pip install -r requirements.txt
```

## Dataset preparation

1. Follow [this repo](https://github.com/qiaosungithub/celebahq256.git) and get dataset in `../data/celeba/celeba-lmdb`.

2. Sanity, using `pytorch_fid/compute_dataset_stat.py`.

```bash
python pytorch_fid/compute_dataset_stat.py \
  --dataset celeba_256 --datadir ../data/celeba/celeba-lmdb \
  --image_size 256 --save_path pytorch_fid/sqa.npy
```

3. Calculate FID between our calculated fid with reference.

```.bash
torchrun --standalone --nproc_per_node=1 fid.py sqa --edm_path=pytorch_fid/celebahq_stat.npy \
    --sqa_ref=pytorch_fid/sqa.npy
```

you may get some number like 0.19.

## Training

```.bash
bash bash_scripts/run.sh
```

Simply comment/uncomment the relevant commands and run `bash run.sh`.

## Testing

### Pretrained model

`https://drive.google.com/file/d/1AIuMr5Ewti6_wQAJdM9elsrERwrxI9Sb/view`

### Sampling

<!-- 1. first check that there is checkpoint at `saved_info/latent_flow/celeba_256/celeb256_f8_adm/model_450.pth`

2. run
```.bash
bash bash_scripts/run_test.sh test_args/celeb256_adm.txt
``` -->

Here is a bug (shape), run the one below!!!!!!

<!-- > Only 1 gpu is required. -->

#### Note (LFM)

To use fixed-steps solver (e.g. `euler` and `heun`), please add `--use_karras_samplers` and change two arguments as follow:

```
METHOD=heun
STEPS=50
```

### Evaluation

1. first check that there is checkpoint at `saved_info/latent_flow/celeba_256/celeb256_f8_adm/model_450.pth`

2. run
```.bash
bash bash_scripts/run_test_ddp.sh test_args/celeb256_adm.txt
```

Here we can use multiple gpus, change this in [bash_scripts/run_test_ddp.sh](./bash_scripts/run_test_ddp.sh).
