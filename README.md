This repo builds off https://github.com/fuyuan-li/tensorflow-RIMs by fuyuan-li

## Install TF-AGENTS

Install https://github.com/tensorflow/agents/ version r0.7.1 under a virtual enviorment with package versions:

- tensorflow-probability 0.12.1

- tensorflow 2.4.0

- dm-env 1.4

- gym 0.18.0

- mujoco_py 2.0.2.13

- tensorflow-estimator 2.4.0  

- dm-tree 0.1.5 

- dm-control 0.0.355168290


## Modify TF agents installation with these updated files

These steps will run the PPO algorithm.

AGENTS_PATH refers to your instllation of tf-agents 0.7.1 .

1. Move __init__.py to /AGENTS_PATH/networks

2. Move lstm_encoding_network.py to /AGENTS_PATH/networks

3. Move LSTM_cell_test.py to /AGENTS_PATH/networks

4. Move GroupLinearLayer.py to /AGENTS_PATH/networks

5. Move GroupLSTMCell.py to /AGENTS_PATH/networks

6. Move RIMCell.py to /AGENTS_PATH/networks

7. Move RIMCellPlay.py to /AGENTS_PATH/networks

9. Move actor_rnn_network.py to /AGENTS_PATH/agents/ddpg

10. BSUITE implementation to follow

## To test

cd /AGENTS_PATH/agents/ppo/examples/v2

python train_eval_clip_agent.py --root_dir="."
