# Video Generators Are Robot Policies
### Arxiv 2025
### [Project Page](https://videopolicy.cs.columbia.edu/) | [Paper](https://videopolicy.cs.columbia.edu/assets/video_policy.pdf) | [ArXiv](https://arxiv.org/abs/2508.00795)

[Junbang Liang](https://junbangliang.github.io/)<sup>1</sup>, [Pavel Tokmakov](https://pvtokmakov.github.io/home/)<sup>2</sup>, [Ruoshi Liu](https://ruoshiliu.github.io/)<sup>1</sup>, [Sruthi Sudhakar](https://sruthisudhakar.github.io/)<sup>1</sup>, [Paarth Shah](https://www.paarthshah.me/)<sup>2</sup>, [Rares Ambrus](https://www.tri.global/about-us/dr-rares-ambrus/)<sup>2</sup>, [Carl Vondrick](https://www.cs.columbia.edu/~vondrick/)<sup>1</sup>

<sup>1</sup>Columbia University, <sup>2</sup>Toyota Research Institute

<p align="center">
  <img src="assets/github_teaser.gif" width="80%">
</p>

##  Usage
###  🛠️ Install Dependencies

Create environment:
```
git clone https://github.com/cvlab-columbia/videopolicy.git
conda create -n videopolicy python=3.10
conda activate videopolicy
```
Install simulation environment:
```
cd packages && \
git clone -b robocasa https://github.com/ARISE-Initiative/robomimic && pip install -e robomimic && \
git clone https://github.com/ARISE-Initiative/robosuite && pip install -e robosuite && \
git clone https://github.com/robocasa/robocasa && pip install -e robocasa && \
python robocasa/robocasa/scripts/download_kitchen_assets.py && \
python robocasa/robocasa/scripts/setup_macros.py
```
Install python packages:
```
cd ..
pip install -r requirements.txt
```

### 🧾 Download Checkpoints and Datasets

Download pretrained checkpoints and move `checkpoints` under the `video_model` folder:
```
wget https://videopolicy.cs.columbia.edu/assets/checkpoints.zip
```
Download simulation dataset and move `datasets` under the `video_model` folder:
```
wget https://videopolicy.cs.columbia.edu/assets/datasets.zip
```

### 🖥️ Run Evaluation

After downloading the pretrained checkpoints amd the simulation dataset, you can run the Robocasa evaluations from the `video_model` folder:

```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/sampling/robocasa_experiment.py --config=scripts/sampling/configs/svd_xt.yaml
```
This will run evaluation on one of the 24 tasks defined in `svd_xt.yaml`. To run on another task, please run this command again on a different gpu.

### 🚀 Run Training

After downloading the pretrained checkpoints amd the simulation dataset, you can run the stage 1 video model training on Robocasa simualtion dataset from the `video_model` folder:

```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --base=configs/stage_1_video_model_training.yaml --name=ft1 --seed=24 --num_nodes=1 --wandb=1 lightning.trainer.devices="0,1,2,3,4,5,6,7"
```

Alternatively, you can run the stage 2 action decoder training with the video model frozen from a pretrained checkpoint, or you can modify `stage_2_action_decoder_training.yaml` to train from your stage 1 checkpoints:

```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --base=configs/stage_2_action_decoder_training.yaml --name=ft1 --seed=24 --num_nodes=1 --wandb=1 lightning.trainer.devices="0,1,2,3,4,5,6,7"
```

Also, we provide an example training the video model and action decoder jointly:

```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --base=configs/joint_training.yaml --name=ft1 --seed=24 --num_nodes=1 --wandb=1 lightning.trainer.devices="0,1,2,3,4,5,6,7"
```

Note that this training script is set for an 8-GPU system, each with 80GB of VRAM. Training with an overall batch size of 32 is found to produce good results, and larger batch size tends to improve model performance.


## 🙏 Acknowledgement
This repository is based on [Stable Video Diffusion](https://github.com/Stability-AI/generative-models) and [Generative Camera Dolly](https://gcd.cs.columbia.edu/). We would like to thank the authors of these work for publicly releasing their code. 

This research is based on work partially supported by the Toyota Research Institute and the NSF NRI Award #2132519.


##  Citation
```
@article{liang2025video,
  title={Video Generators are Robot Policies}, 
  author={Liang, Junbang and Tokmakov, Pavel and Liu, Ruoshi and Sudhakar, Sruthi and Shah, Paarth and Ambrus, Rares and Vondrick, Carl},
  journal={arXiv preprint arXiv:2508.00795},
  year={2025}
}
```
