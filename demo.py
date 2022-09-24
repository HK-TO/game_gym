import time
import gym
import dogfight
import pygame

import numpy as np

# pygame.init()
# screen = pygame.display.set_mode((400, 300))
# pygame.display.set_caption("keyboard event")
# import sys
# while True:

#     screen.fill((0, 0, 0))
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_ESCAPE:
#                 pygame.quit()
#                 sys.exit()
            
#             else:
#                 print("押されたキー = " + pygame.key.name(event.key))
#         pygame.display.update()


from check_lander import LunarLander
env = gym.make('LunarLanderContinuous-v2', render_mode="human")

# env = dogfight.DogFight(render_mode="human")
# env = LunarLander(render_mode="human")
num_episodes = 10

for i in range(num_episodes):
    state = env.reset()
    
    totalReward = 0

    for _ in range(100):
        action = np.zeros(2, dtype=np.float32)
        keystate = pygame.key.get_pressed()
        # for event in pygame.event.get():
        #     print(event)
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
        observation = env.step(randomAction)
        print(observation)
        # observation,reward,done,info = env.step(randomAction) 
        # observation,reward,done,info = env.step(action) 

        time.sleep(0.1)
        # totalReward += reward

    print('Episode', i,', Total reward:', totalReward)

env.close()
