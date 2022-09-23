import imp
import time
import gym
import dogfight
import pygame

import numpy as np

pygame.init()

# env = gym.make('CarRacing-v0')
env = dogfight.DogFight()
num_episodes = 10
for i in range(num_episodes):
    state = env.reset()
    totalReward = 0

    for _ in range(100):
        action = np.zeros(2, dtype=np.float32)
        keystate = pygame.key.get_pressed()
        for event in pygame.event.get():
            print(event.type)
            print(pygame.K_BACKQUOTE)
        if keystate[pygame.K_UP]:
            action[1] += 1.0
        if keystate[pygame.K_DOWN]:
            action[1] -= 1.0
        if keystate[pygame.K_RIGHT]:
            action[0] += 1.0
        if keystate[pygame.K_LEFT]:
            action[0] -= 1.0

        env.render()

        # take a random action
        randomAction = env.action_space.sample()
        # observation,reward,done,info = env.step(randomAction) 
        observation,reward,done,info = env.step(action) 

        time.sleep(0.1)
        totalReward += reward

    print('Episode', i,', Total reward:', totalReward)

env.close()
