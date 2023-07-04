# Downstream tasks

## Training

For downstream tasks as image inpaiting and semantic synthesis, we use the below commands.

<details>
<summary>Image inpainting</summary>

```
python train_flow_latent_inpainting.py --exp inpainting_kl --dataset celeba_256 \
  --batch_size 64 --lr 5e-5 --scale_factor 0.18215 --num_epoch 500 --image_size 256 \
  --num_in_channels 9 --num_out_channels 4 --ch_mult 1 2 3 4 --attn_resolution 16 8 \
  --num_process_per_node 2 --save_content
```

</details>

<details>
<summary>Semantic Synthesis</summary>

```
python train_flow_latent_semantic_syn.py --exp semantic_kl --dataset celeba_256 \
  --batch_size 64 --lr 5e-5 --scale_factor 0.18215 --num_epoch 175 --image_size 256 \
  --num_in_channels 8 --num_out_channels 4 --ch_mult 1 2 3 4 --attn_resolution 16 8 \
  --num_process_per_node 2 --save_content
```

</details>

## Testing

For downstream tasks, we firstly run `test_flow_latent_semantic_syn.py` and `test_flow_latent_inpainting.py` to generate the synthesis data based on given conditions. After that, we can evaluate the metric using below commands.

**Image Inpainting**

```
# generate the image from inpainting data
python test_flow_latent_inpainting.py --exp inpainting_kl --dataset celeba_256 \
  --batch_size 64 --lr 5e-5 --scale_factor 0.18215 --num_epoch 500 --image_size 256 \
  --num_in_channels 9 --num_out_channels 4 --ch_mult 1 2 3 4 --attn_resolution 16 8 \

python pytorch_fid/cal_inpainting.py <path_to_generated_data> <path_to_gt_data>
```

**Semantic Synthesis**

```
# generate the image from segmentation map
python test_flow_latent_semantic_syn.py --exp semantic_kl --dataset celeba_256 \
  --batch_size 64 --lr 5e-5 --scale_factor 0.18215 --num_epoch 500 --image_size 256 \
  --num_in_channels 8 --num_out_channels 4 --ch_mult 1 2 3 4 --attn_resolution 16 8 \

python pytorch_fid/fid_score.py <path_to_generated_data> <path_to_gt_data>
```