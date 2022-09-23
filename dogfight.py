from traceback import print_tb
import gym
import numpy as np
import cv2

class DogFight(gym.Env):
    def __init__(self):
        self.WINDOW_SIZE = 600 #画面サイズの決定
        self.ACTION_MAP = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]) #アクションの用意
        self.GOAL_RANGE = 50 #ゴールの範囲設定

        # アクション数定義
        self.num_actions = 2
        self.action_space = gym.spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        # 状態の範囲を定義
        self.num_states = 2
        self.observation_space = gym.spaces.Box(np.zeros(self.num_states), np.ones(self.num_states) * np.Inf)

        self.reset()

    def reset(self):
        self.timestep = 0

        # ボールとゴールの位置をランダムで初期化
        self.ball_position = np.random.rand(self.num_states) * float(self.WINDOW_SIZE)
        self.goal_position = np.random.rand(self.num_states) * float(self.WINDOW_SIZE)

        # 状態の作成
        vec = self.ball_position - self.goal_position
        observation = np.arctan2(vec[0], vec[1]) # 角度の計算
        observation = np.array([observation])

        self.before_distance = np.linalg.norm(vec)

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
        # opencvで描画処理してます
        img = np.zeros((self.WINDOW_SIZE, self.WINDOW_SIZE, 3)) #画面初期化
        goal_position = self.goal_position.astype(np.int32)
        ball_position = self.ball_position.astype(np.int32)

        cv2.circle(img,  tuple(goal_position), 10, (0, 255, 0), thickness=-1) #ゴールの描画
        cv2.circle(img, tuple(goal_position), self.GOAL_RANGE, color=(0,255,0), thickness=5) #ゴールの範囲の描画

        cv2.circle(img,  tuple(ball_position), 10, (0, 0, 255), thickness=-1) #ボールの描画

        cv2.imshow('image', img)
        cv2.waitKey(1)