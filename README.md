## Medical vision foundation models for clinical images

Official repo including a series of DINOv3-based medical vision foundation models.<br>
`[RETFound-MAE]`:[RETFound: a foundation model for generalizable disease detection from retinal images](https://www.nature.com/articles/s41586-023-06555-x).<br>
`[RETFound-DINOv2]`:[Revealing the Impact of Pre-training Data on Medical Foundation Models](https://www.nature.com/articles/s41467-026-70077-z).<br>
`[RETFound-DINOv3]`:[Generalist versus Specialist Vision Foundation Models for Ocular Disease and Oculomics](https://arxiv.org/abs/2509.03421).<br>
`[ChestFound-DINOv3]`:[Generalist versus Specialist Vision Foundation Models for Ocular Disease and Oculomics](https://arxiv.org/abs/2509.03421).<br>

Please contact 	**ykzhoua@gmail.com** or **yukun.zhou.19@ucl.ac.uk** if you have questions.


### 📝Key features

- RETFound-DINOv3 shows competitive performance and best efficiency in retinal image analysis, including classification, regression, and segmentation.
- ChestFound-DINOv3 achieved best data efficiency and calibration in respiratory disease detection.


### 🎉News

- 🐉2026/04: **Preprint, code, and model weights online [available](https://arxiv.org/abs/2509.03421)!**


### 🔧Install environment

1. Create environment with conda:

```
conda create -n medfound python=3.11.0 -y
conda activate medfound
```

2. Install dependencies

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/HORIZONHealthcare/Med_DINOv3
cd Med_DINOv3
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name medfound --display-name "Python (medfound)"
```


### 🌱Fine-tuning with Medfound weights

1. Get access to the pre-trained models on HuggingFace (register an account and fill in the form) and go to step 2:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">Source</th>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_natureCFP</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_natureCFP">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41586-023-06555-x">Nature RETFound paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_natureOCT</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_natureOCT">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41586-023-06555-x">Nature RETFound paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_meh</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_meh">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41467-026-70077-z">FM data paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_shanghai</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_shanghai">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41467-026-70077-z">FM data paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_dinov2_meh</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_dinov2_meh">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41467-026-70077-z">FM data paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_dinov2_shanghai</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_dinov2_shanghai">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41467-026-70077-z">FM data paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_dinov3</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_dinov2_shanghai">access</a></td>
<td align="center"><a href="https://arxiv.org/abs/2509.03421">Benchmark paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">ChestFound_dinov3</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_dinov2_shanghai">access</a></td>
<td align="center"><a href="https://arxiv.org/abs/2509.03421">Benchmark data paper</a></td>
</tr>
</tbody></table>

1. Login in your HuggingFace account, where HuggingFace token can be [created and copied](https://huggingface.co/settings/tokens).
```
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN
```

**Optional**: if your machine and server cannot access HuggingFace due to internet wall, run the command below (Do not run it if you can access):
```
export HF_ENDPOINT=https://hf-mirror.com
```

3. Organise your data into this directory structure (Public datasets used in this study can be [downloaded here](BENCHMARK.md))

```
├── data folder
    ├──train
        ├──class_a
        ├──class_b
        ├──class_c
    ├──val
        ├──class_a
        ├──class_b
        ├──class_c
    ├──test
        ├──class_a
        ├──class_b
        ├──class_c
``` 


5. Start fine-tuning by running `sh train.sh`.


In `train.sh`, the model can be selected by changing the hyperparameters `MODEL`, `MODEL_ARCH`, `FINETUNE`:

**RETFound**:

| MODEL           | MODEL_ARCH               | FINETUNE                 | SIZE                     |
|-----------------|--------------------------|--------------------------|--------------------------|
| RETFound_dinov2 | retfound_dinov2          | RETFound_dinov2_meh      | ~300M                    |
| RETFound_dinov2 | retfound_dinov2          | RETFound_dinov2_shanghai | ~300M                    |
| MED_dinov3 | MED_dinov3          | retfound_dinov3 | ~300M                    |


**ChestFound**:

| MODEL           | MODEL_ARCH               | FINETUNE                 | SIZE                     |
|-----------------|--------------------------|--------------------------|--------------------------|
| MED_dinov3 | MED_dinov3          | chestfound_dinov3 | ~300M                    |


**DINOv3**:

| MODEL           | MODEL_ARCH               | FINETUNE                         | SIZE                     |
|-----------------|--------------------------|----------------------------------|--------------------------|
| Dinov3          | dinov3_vits16            | dinov3_vits16_pretrain.pth       | ~21M                     |
| Dinov3          | dinov3_vits16plus        | dinov3_vits16plus_pretrain.pth   | ~29M                     |
| Dinov3          | dinov3_vitb16            | dinov3_vitb16_pretrain.pth       | ~86M                     |
| Dinov3          | dinov3_vitl16            | dinov3_vitl16_pretrain.pth       | ~300M                    |
| Dinov3          | dinov3_vith16plus        | dinov3_vith16plus_pretrain.pth   | ~840M                    |
| Dinov3          | dinov3_vit7b16           | dinov3_vit7b16_pretrain.pth      | ~6.7B                    |


**DINOv2**:

| MODEL           | MODEL_ARCH               | FINETUNE                     | SIZE                     |
|-----------------|--------------------------|------------------------------|--------------------------|
| Dinov2          | dinov2_vits14            | dinov2_vits14_pretrain.pth   | ~21M                     |
| Dinov2          | dinov2_vitb14            | dinov2_vitb14_pretrain.pth   | ~86M                     |
| Dinov2          | dinov2_vitl14            | dinov2_vitl14_pretrain.pth   | ~300M                    |
| Dinov2          | dinov2_vitg14            | dinov2_vitg14_pretrain.pth   | ~1.1B                    |


Change the DATASET to your dataset directory.

```
# ==== Model settings ====
# adaptation {finetune,lp}
ADAPTATION="finetune"
MODEL="MED_dinov3"
MODEL_ARCH="MED_dinov3"
FINETUNE="chestfound_dinov3"

# ==== Data settings ====
DATASET="tbx11k"
DATA_PATH="./${DATASET}"
NB_CLASSES=3

# ==== Training settings ====
MASTER_PORT=48788
EPOCHS=50
BATCH_SIZE=24
INPUT_SIZE=224
DATA_RATIO="1.0"

TASK="${MODEL_ARCH}_${DATASET}_${BATCH_SIZE}_${ADAPTATION}_${DATA_RATIO}"

torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" main_finetune.py \
  --model "${MODEL}" \
  --model_arch "${MODEL_ARCH}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --nb_classes "${NB_CLASSES}" \
  --data_path "${DATA_PATH}" \
  --input_size "${INPUT_SIZE}" \
  --dataratio "${DATA_RATIO}" \
  --task "${TASK}" \
  --adaptation "${ADAPTATION}" \
  --finetune "${FINETUNE}"

```



6. For evaluation only (download data and model checkpoints [here](BENCHMARK.md); change the DATA_PATH below)


```
# ==== Model settings ====
# adaptation {finetune,lp}
ADAPTATION="finetune"
MODEL="MED_dinov3"
MODEL_ARCH="MED_dinov3"
FINETUNE="chestfound_dinov3"

# ==== Data settings ====
DATASET="tbx11k"
DATA_PATH="./${DATASET}"
NB_CLASSES=3

# ==== Evaluation settings ====
MASTER_PORT=48788
EPOCHS=50
BATCH_SIZE=24
INPUT_SIZE=224
DATA_RATIO="1.0"

TASK="${MODEL_ARCH}_${DATASET}_${BATCH_SIZE}_${ADAPTATION}_${DATA_RATIO}"

# Path to the trained checkpoint (adjust if you saved elsewhere)
CKPT="./output_dir/${TASK}/checkpoint-best.pth"


torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" main_finetune.py \
  --model "${MODEL}" \
  --model_arch "${MODEL_ARCH}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --nb_classes "${NB_CLASSES}" \
  --data_path "${DATA_PATH}" \
  --input_size "${INPUT_SIZE}" \
  --dataratio "${DATA_RATIO}" \
  --task "${TASK}" \
  --adaptation "${ADAPTATION}" \
  --finetune "${FINETUNE}" \
  --eval \
  --resume "${CKPT}"

```


### 📃Citation

If you find this repository useful, please consider citing this paper:


```
@article{zhou2023foundation,
  title={A foundation model for generalizable disease detection from retinal images},
  author={Zhou, Yukun and Chia, Mark A and Wagner, Siegfried K and Ayhan, Murat S and Williamson, Dominic J and Struyven, Robbert R and Liu, Timing and Xu, Moucheng and Lozano, Mateo G and Woodward-Court, Peter and others},
  journal={Nature},
  volume={622},
  number={7981},
  pages={156--163},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

```
@article{zhou2026understanding,
  title={Understanding pre-training data effects in retinal foundation models using two large fundus cohorts},
  author={Zhou, Yukun and Wang, Zheyuan and Wu, Yilan and Ong, Ariel Yuhan and Wagner, Siegfried K and Ruffell, Eden and Chia, Mark A and Guan, Zhouyu and Ju, Lie and Engelmann, Justin and others},
  journal={Nature Communications},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

```
@misc{zhou2025generalistversusspecialistvision,
      title={Generalist versus Specialist Vision Foundation Models for Ocular Disease and Oculomics}, 
      author={Yukun Zhou and Paul Nderitu and Jocelyn Hui Lin Goh and Justin Engelmann and Siegfried K. Wagner and Anran Ran and Hongyang Jiang and Lie Ju and Ke Zou and Sahana Srinivasan and Hyunmin Kim and Takahiro Ninomiya and Zheyuan Wang and Gabriel Dawei Yang and Eden Ruffell and Dominic Williamson and Rui Santos and Gabor Mark Somfai and Carol Y. Cheung and Tien Yin Wong and Daniel C. Alexander and Yih Chung Tham and Pearse A. Keane},
      year={2025},
      eprint={2509.03421},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2509.03421}, 
}
```
