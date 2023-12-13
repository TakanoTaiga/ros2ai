# ros2ai
AI Model for ROS 2.

## Env Setup
### Requestments
- ROS 2
- CUDA & cuDNN
- torch
- transformers
- accelerate

### Easy setup
```bash
sudo apt install git curl
git clone https://github.com/autowarefoundation/autoware.git -b main --single-branch
cd autoware
./setup-dev-env.sh
```
```bash
python3 -m pip install -U torch torchvision torchaudio git+https://github.com/huggingface/transformers accelerate
```

## How To Use

### Image-To-Text

<details>
<summary>blip_image_captioning_base</summary>

```bash
ros2 run ros2ai blip_image_captioning_base
```
</details>

<details>
<summary>blip_image_captioning_large</summary>

```bash
ros2 run ros2ai blip_image_captioning_large
```
</details>

<details>
<summary>pix2struct_textcaps_base</summary>

```bash
ros2 run ros2ai pix2struct_textcaps_base
```
</details>

<details>
<summary>vit_gpt2_image_captioning</summary>

```bash
ros2 run ros2ai vit_gpt2_image_captioning
```
</details>

### Vision QA
<details>
<summary>vilt_b32_finetuned_vqa</summary>

```bash
ros2 run ros2ai vilt_b32_finetuned_vqa
```
</details>


### Image-classification

<details>
<summary>resnet_50</summary>

```bash
ros2 run ros2ai resnet_50
```
</details>