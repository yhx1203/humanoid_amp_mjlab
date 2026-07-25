# humanoid_amp_mjlab

## Overview

`humanoid_amp_mjlab` is a reinforcement learning codebase built on MJLab for AMP-based Unitree G1 humanoid locomotion.

## Installation
**Conda environment**

```bash
conda create -n mjlab python=3.11
conda activate mjlab
```

**Install dependencies**
```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

**Install humanoid_amp_mjlab**
```bash
git clone https://github.com/yhx1203/humanoid_amp_mjlab
```

```bash
cd humanoid_amp_mjlab
pip install -e .
```


## Training

You can browse the motion set before training.

```bash
python scripts/view_csv_in_mujoco.py src/assets/motions/g1_seed/arc_walk_left_loop_001__A037_M.csv \
  --once 
```
<!-- ![replay](docs/replay.gif) -->


```bash
python scripts/train.py Unitree-G1-AMP-Flat \
  --env.scene.num-envs 4096 \
  --agent.run-name amp_g1_seed \
  --agent.upload-model False
```

## Evaluate
```bash
python scripts/play.py Unitree-G1-AMP-Flat \
  --checkpoint-file logs/rsl_rl/g1_amp_walking/amp_seed/model_2900.pt \
  --num-envs 1 \
  --viewer viser
```
<!-- ![viser](docs/viser.gif) -->

## Sim2sim

**Preparation**

[unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)

```bash
conda activate mjlab

cd ~
sudo apt install python3-pip
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```


**Terminal 1**

```bash
cd humanoid_amp_mjlab
conda activate mjlab
python deploy/amp/sim2sim/g1_mjlab_simulator.py \
  --joystick \
  --joystick-type xbox \
  --joystick-device 0
```

**Terminal 2**
```bash
cd humanoid_amp_mjlab
conda activate mjlab

python deploy/amp/sim2sim/g1_amp_sim2sim.py \
  --checkpoint-file logs/rsl_rl/g1_amp_walking/amp_seed/model_2900.pt \
  --wireless 
```

<!-- ![mujoco](docs/mujoco.gif) -->



## Acknowledgements

This project builds upon and benefits from the following open-source repositories:

- [mjlab](https://github.com/mujocolab/mjlab)
- [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)
- [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
- [bones-studio](https://huggingface.co/datasets/bones-studio/seed)


 

