# DeepLearningStepbyStep

### This is a not modular version of each of the beginner algorithm of deep learning 

## 03. Q-Learning with Epsilon decay, the average score is 7.76 with 100 episodes
  
 <img width="698" alt="Screenshot 2023-08-17 at 14 52 45" src="https://github.com/dada325/DeepLearningStepbyStep/assets/7775973/445db7d8-3643-4f57-b8dc-9c914475283b">


## 04. Deep Q-Learning testing with env Cartpole, still not stable

  <img width="657" alt="Screenshot 2023-08-17 at 14 55 55" src="https://github.com/dada325/DeepLearningStepbyStep/assets/7775973/78a75126-3f3d-4f62-8682-80ca84658b27">

## 05. DDPG testing with Pendulumn, 


<img width="710" alt="Screenshot 2023-08-17 at 16 10 05" src="https://github.com/dada325/DeepLearningStepbyStep/assets/7775973/8593bae2-fe8f-48e1-84f6-366d39ecfbf6">




In the `Pendulum-v1` environment, the best possible reward is 0. 
Here are some thoughts to enhance the performance of DDPG on `Pendulum-v1`:

1. **More Training**: DDPG typically requires a lot of training to converge stably. Try increasing the training episodes, perhaps to 1000 or more.

2. **Network Architecture & Hyperparameters**: Experiment with different network architectures or adjust hyperparameters, like learning rates, optimizers, discount factors, soft update parameters, etc. Hyperparameter tuning is common and often necessary in deep reinforcement learning.

3. **Exploration**: DDPG uses a deterministic policy, but to boost exploration, we added some noise. Try different types and magnitudes of noise to enhance performance.

4. **Replay Buffer**: Ensure to have a sufficiently large replay buffer and use uniform sampling. Consider using priority replay, where more critical transitions have a higher chance of being sampled.

5. **Target Network Update Frequency**: Try updating the target networks more frequently or less often.

6. **Learning Rate Scheduling**: Using different learning rates at different stages of training could be beneficial. For instance, start with a higher learning rate and gradually decrease it over time.


When tweaking the model, change only one parameter and then evaluate performance, so we can better understand the impact of each adjustment.

now , I will try TD3 and SAC. 
