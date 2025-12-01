# Balatro-Gym
A Gymnasium environment for the game Balatro for use with reinforcement learning (RL) libraries such as TorchRL and Stable Baselines3. Also includes RL training and evaluation scripts. 

# Demonstration
In these videos, the policy is performing rollouts of game hands to learn from. All gameplay is completely automated on a live game client using the [balatrobot API](https://github.com/coder/balatrobot) and requires no human input. Full training speed (512x) to accelerate rollouts is accomplished using the [HandyBalatro](https://github.com/SleepyG11/HandyBalatro) mod. 
## Human Speed Training Loop
https://github.com/user-attachments/assets/c14d8e2f-2f89-4fa1-9390-8d8d3050b259
## Full Speed Training Loop
https://github.com/user-attachments/assets/f6661b27-cb83-42bc-9020-35ea6025d90c

# Evaluation
Coming soon

# Installation
Coming soon

# Usage
Coming soon

# Requirements
- BalatroBot: https://coder.github.io/balatrobot/installation/


# Todo
- [ ] Improve observation space definition to account for complex hand space embedding. 
- [ ] Add engineered features to observation space such as `has_flush`. 
- [ ] Add basic features to observation space such as `hands_left`. 
- [ ] Refactor network architecture to include multi-head output for shop decisions.
- [ ] Improve README Evaluation, Installation, Usage, and Requirements sections.
- [ ] 
The project has an associated feature board that you can view [here](https://github.com/users/Jonathanace/projects/3). 
