# CLIP for image-text retrieval 

>End of studies internship project


*This repository is a fork of [mlfundation implementation](https://github.com/mlfoundations/open_clip) to train and use the CLIP model, please check their [README](ml_fundation_doc/mlfund_README.md) as well as this one!*

This aims at delivering a way to use and to finetune the CLIP model. 

## 🪛 Installation

```bash
docker build . -t open-clip
```

```bash
docker run --rm -ti -v ${PWD}:/home/open-clip -v /my_dataset:/home/open-clip/my_dataset open-clip:latest
```

add `--gpus '"device=0,1,2,3"`  to use gpus 

add  `-v /dev/shm:/dev/shm` if you are training on multiple gpus (this will enable access to shared memory)

⚠ This project uses python 3.7. 

## 🍄 Usage
### Metrics 
Each of these metrics is available from image to text, and from text to image. 
- Median rank 
- Mean rank 
- R@X accuracy : accuracy on : 'grounth truth in the top-X ranked answers ?'

### Available models
Open CLIP project tries to reach the same metric presented in OpenAI's paper. You can choose between OpenAI's pre-trained weights and open-clip pre-trained weights, with each available architecture and different pre-trained dataset. 

Use the following commands to list the available models and their weights and to load a model. 
```python
import open_clip
open_clip.list_pretrained()
model, train_transform, eval_transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_e16')
```
To load other pre-trained image use `pretrained-image`
```python
import open_clip
model, train_transform, eval_transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained-image='my_checkpoint_path')
```

###  👉 Basics 
**Simple inference**

```python
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
text = open_clip.tokenize(["a diagram", "a dog", "a cat"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
```

**Evaluation**
Use the clip_test_acc.py script :
1. Compute simple metrics on all dataset (may not fit GPU!)
```bash
python retrieval.main 
	--ground_truth_csv_path GROUND_TRUTH_CSV_PATH \
    --csv_img_key CSV_IMG_KEY \
    --csv_caption_key CSV_CAPTION_KEY \
    --input_dir INPUT_DIR \
    --csv_separator CSV_SEPARATOR \
    [--network NETWORK] \
    [--checkpoint CHECKPOINT] \
    [--workers WORKERS] \
    [--device DEVICE] \
    [--pretrained PRETRAINED] \
    [--log_rate LOG_RATE] \
    [--tops TOPS] \
```

where 
- `ground_truth_csv_path` is the csv where you store the image filename, its label and shooting id
- `csv_img_key` is the name of the column where the filename are 
- `csv_caption_key` is the name of the column where the labels are 
- `input_dir` is the folder where the images are stored
- `csv_separator` is the separator character of your csv file 
- `network` is the name of the network, see [Available Models](###Available-models) for more precisions
- `checkpoint` is the filename of the checkpoint 
- `workers` are the number of workers 
- `device` is `cpu` or `cuda` , default is `cpu`
- `pretrained` is the source of the pretrained model, see [Available Models](###Available-models) for more precisions
- `log_rate` is the rate for printing the metrics, default is 10
- `tops` is the accuracy tops to compute, to enter with spaces (e.g `1 2 4 9`), default `1 2 3 5 10`

2. Compute average metrics on shootings 
	add `--per_shooting --csv_shooting_key CSV_SHOOTING_KEY` to retrieval command

3. From the training main 
from local checkpoint : 
```bash 
python -m training.main
    --val-data="/path/to/validation_data.csv"  
    --model RN101 
    --pretrained /path/to/checkpoints/epoch_K.pt
```
from a hosted pretrained checkpoint
```bash
python -m training.main 
    --imagenet-val /path/to/imagenet/validation 
    --model ViT-B-32-quickgelu 
    --pretrained laion400m_e32
```

### 👉 Training
All the parameters can be found in `training/params.py` 

**Single GPU** (example)
```bash
python -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="path to train data csv" \
	--val-data="path to validation csv" \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=/path/to/imagenet/root/val/ \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --model RN50
```

**Multi GPUs** (example)
```bash

torchrun --nproc_per_node 4 -m training.main \
	--train-data="path to train data csv" \
	--val-data="path to validation csv" \
	--csv-img-key "new filename" \
	--csv-caption-key "food label" \
	--csv-separator ',' \
	--batch-size 128 \
	--precision amp \
	--workers 4 \
	--model ViT-B-32 \
	--epochs=40 \ 
	--save-frequency 15 \
	--pretrained 'openai' \
	--warmup 100 \
	--lr 5.0e-5\
	--val-frequency 2
```

**🔒 LiT**

LiT consist in lock the image tower and unlock the text tower. open-clip offers parameters to use this technique to fine-tune CLIP. 
Use the following parameters : 
- `--lock-image` to lock full image tower by disabling gradients.
- `--lock-image-unlocked-groups n`  to leave last n image tower layer groups unlocked.
- `--lock-image-freeze-bn-stats` to freeze BatchNorm running stats in image tower for any locked layers

**Weight and Biases**
- Log to weight and biases with `wandb login`
- Add  `--report-to 'wandb'` in script parameters 
- Open your WandB dashboard, you're set ! 

## 🌶 Dataset tools 
Some script are available inside `src/data` for dataset management

`gather_cc.py` is an open-clip tool to download conceptual caption dataset. 



## 🔗 Resources
**Articles** 
- [CLIP](https://openai.com/blog/clip/) , [article](https://arxiv.org/abs/2103.00020), [original code](https://github.com/openai/CLIP) 
- LiT, Zero-Shot Transfer with Locked-image text Tuning, [article](https://arxiv.org/abs/2111.07991), [code](https://github.com/google-research/vision_transformer#lit-models)

**Repositories**
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Open-CLIP from ML fundation](https://github.com/mlfoundations/open_clip)
