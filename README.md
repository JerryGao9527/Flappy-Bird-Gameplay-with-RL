# Flappy-Bird-Gameplay-with-RL
Create a conda environment with the required packages using the following terminal commands:
- conda create -n Flappy_Bird -c conda-forge -y python=3.9
- conda activate Flappy_Bird
- pip install -r requirements.txt

python flappy_bird_gymnasium/PG_REINFORCE.py --num_episodes 100000 --batch_size 16 --lr 0.0001 --gamma 0.95 --epsilon 0.9


Please see DQN implementation at:

https://github.com/leocheung1001/flappy_bird_rl?tab=readme-ov-file
