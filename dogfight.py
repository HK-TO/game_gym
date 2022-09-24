from typing import TYPE_CHECKING, Optional

import gym
from gym.error import DependencyNotInstalled

import numpy as np
import cv2

if TYPE_CHECKING:
    import pygame

FPS = 50

VIEWPORT_W = 600
VIEWPORT_H = 600


class DogFight(gym.Env):

    metadeta = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,):
        self.ACTION_MAP = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]) #アクションの用意
        self.GOAL_RANGE = 50 #ゴールの範囲設定

        # アクション数定義
        self.num_actions = 2
        self.action_space = gym.spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        # 状態の範囲を定義
        self.num_states = 2
        self.observation_space = gym.spaces.Box(np.zeros(self.num_states), np.ones(self.num_states) * np.Inf)

        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True

        self.render_mode = render_mode

    def reset(self):
        self.timestep = 0

        # ボールとゴールの位置をランダムで初期化
        self.ball_position = np.random.rand(self.num_states) * float(VIEWPORT_W)
        self.goal_position = np.random.rand(self.num_states) * float(VIEWPORT_H)

        # 状態の作成
        vec = self.ball_position - self.goal_position
        observation = np.arctan2(vec[0], vec[1]) # 角度の計算
        observation = np.array([observation])

        self.before_distance = np.linalg.norm(vec)

        if self.render_mode == "human":
            self.render()

        return observation

    def step(self, action):
        # print(action)
        # print(self.ball_position)
        self.ball_position = self.ball_position-action

        # 状態の作成
        vec = self.ball_position - self.goal_position
        observation = np.arctan2(vec[0], vec[1]) # 角度の計算
        observation = np.array([observation])

        # 報酬の計算
        distance = np.linalg.norm(vec) # 距離の計算
        reward = self.before_distance - distance # どれだけゴールに近づいたか

        # 終了判定
        done = False
        if distance < self.GOAL_RANGE:
            done = True

        self.before_distance = distance

        return observation, reward, done, {}

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[box2d]`"
            )

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.transform.scale(self.surf, (30, 30))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        # for obj in self.particles:
        #     obj.ttl -= 0.15
        #     obj.color1 = (
        #         int(max(0.2, 0.15 + obj.ttl) * 255),
        #         int(max(0.2, 0.5 * obj.ttl) * 255),
        #         int(max(0.2, 0.5 * obj.ttl) * 255),
        #     )
        #     obj.color2 = (
        #         int(max(0.2, 0.15 + obj.ttl) * 255),
        #         int(max(0.2, 0.5 * obj.ttl) * 255),
        #         int(max(0.2, 0.5 * obj.ttl) * 255),
        #     )

        # self._clean_particles(False)

        # opencvで描画処理してます
        # img = np.zeros((VIEWPORT_W, VIEWPORT_H, 3)) #画面初期化
        # goal_position = self.goal_position.astype(np.int32)
        # ball_position = self.ball_position.astype(np.int32)

        # cv2.circle(img,  tuple(goal_position), 10, (0, 255, 0), thickness=-1) #ゴールの描画
        # cv2.circle(img, tuple(goal_position), self.GOAL_RANGE, color=(0,255,0), thickness=5) #ゴールの範囲の描画

        # cv2.circle(img,  tuple(ball_position), 10, (0, 0, 255), thickness=-1) #ボールの描画

        # cv2.imshow('image', img)
        # cv2.waitKey(1)

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False