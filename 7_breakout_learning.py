#!/usr/bin/env python
# coding: utf-8

# ## 7章　ブロック崩しBreakoutの学習プログラム    

# In[1]:


# パッケージのimport
import numpy as np
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
# import  gymnasium as gym
from gym import spaces
from gym.spaces.box import Box
import os

CUDA_VISIBLE_DEVICES = 0,1
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# In[2]:


# 実行環境の設定
# 参考：https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

import cv2
cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        '''工夫1のNo-Operationです。リセット後適当なステップの間何もしないようにし、
        ゲーム開始の初期状態を様々にすることｆで、特定の開始状態のみで学習するのを防ぐ'''

        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        print('reset1')
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(
                1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            (obs, _, done, _) = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        print('step1')
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        '''工夫2のEpisodic Lifeです。1機失敗したときにリセットし、失敗時の状態から次を始める'''
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        print("step2")
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        # print("self.lives",self.env.unwrapped.ale.lives())
        print("lives", lives)
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        print("self.lives",self.lives)
        return obs, reward, done, info

    def reset(self, **kwargs):
        print("reset2")
        '''5機とも失敗したら、本当にリセット'''
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        '''工夫3のMax and Skipです。4フレーム連続で同じ行動を実施し、最後の3、4フレームの最大値をとった画像をobsにする'''
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        print("step3")
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        print("reset3")
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        '''工夫4のWarp frameです。画像サイズをNatureのDQN論文と同じ84x84の白黒にします'''
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        '''PyTorchのミニバッチのインデックス順に変更するラッパー'''
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# In[3]:


# 実行環境生成関数の定義

# 並列実行環境
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def make_env(env_id, seed, rank):
    def _thunk():
        '''_thunk()がマルチプロセス環境のSubprocVecEnvを実行するのに必要'''

        env = gym.make(env_id)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env.seed(seed + rank)  # 乱数シードの設定
        env = EpisodicLifeEnv(env)
        env = WarpFrame(env)
        env = WrapPyTorch(env)

        return env

    return _thunk


# In[4]:


# 定数の設定

ENV_NAME = 'BreakoutNoFrameskip-v4'
# ENV_NAME='BreakoutDeterministic-v4'
# Breakout-v0ではなく、BreakoutNoFramesk
# ip-v4を使用
# v0はフレームが自動的に2-4のランダムにskipされますが、今回はフレームスキップはさせないバージョンを使用
# 参考URL https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
# https://github.com/openai/gym/blob/5cb12296274020db9bb6378ce54276b31e7002da/gym/envs/__init__.py#L371
    
NUM_SKIP_FRAME = 4 # skipするframe数です
NUM_STACK_FRAME = 4  # 状態として連続的に保持するframe数です
NOOP_MAX = 30  #  reset時に何もしないフレームを挟む（No-operation）フレーム数の乱数上限です
NUM_PROCESSES =16#  並列して同時実行するプロセス数です
NUM_ADVANCED_STEP = 5  # 何ステップ進めて報酬和を計算するのか設定
GAMMA = 0.99  # 時間割引率

TOTAL_FRAMES=10e6  #  学習に使用する総フレーム数
NUM_UPDATES = int(TOTAL_FRAMES / NUM_ADVANCED_STEP / NUM_PROCESSES)
NUM_UPDATES =NUM_UPDATES *100
# ネットワークの総更新回数
# NUM_UPDATESは125,000となる


# In[5]:


# A2Cの損失関数の計算のための定数設定
value_loss_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5

# 学習手法RMSpropの設定
lr = 7e-4
eps = 1e-5
alpha = 0.99


# In[6]:


# GPUの使用の設定
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[7]:


# メモリオブジェクトの定義


class RolloutStorage(object):
    '''Advantage学習するためのメモリクラスです'''

    def __init__(self, num_steps, num_processes, obs_shape):

        self.observations = torch.zeros(
            num_steps + 1, num_processes, *obs_shape).to(device)
        # *を使うと()リストの中身を取り出す
        # obs_shape→(4,84,84)
        # *obs_shape→ 4 84 84

        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(device)
        self.actions = torch.zeros(
            num_steps, num_processes, 1).long().to(device)

        # 割引報酬和を格納
        self.returns = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        self.index = 0  # insertするインデックス

    def insert(self, current_obs, action, reward, mask):
        '''次のindexにtransitionを格納する'''
        self.observations[self.index + 1].copy_(current_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP  # インデックスの更新

    def after_update(self):
        '''Advantageするstep数が完了したら、最新のものをindex0に格納'''
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        '''Advantageするステップ中の各ステップの割引報酬和を計算する'''

        # 注意：5step目から逆向きに計算しています
        # 注意：5step目はAdvantage1となる。4ステップ目はAdvantage2となる。・・・
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] *                 GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]


# In[8]:


# A2Cのディープ・ニューラルネットワークの構築


def init(module, gain):
    '''層の結合パラメータを初期化する関数を定義'''
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module


class Flatten(nn.Module):
    '''コンボリューション層の出力画像を1次元に変換する層を定義'''

    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self, n_out):
        super(Net, self).__init__()

        # 結合パラメータの初期化関数
        def init_(module): return init(
            module, gain=nn.init.calculate_gain('relu'))

        # コンボリューション層の定義
        self.conv = nn.Sequential(
            # 画像サイズの変化84*84→20*20
            init_(nn.Conv2d(NUM_STACK_FRAME, 32, kernel_size=8, stride=4)),
            # stackするflameは4画像なのでinput=NUM_STACK_FRAME=4である、出力は32とする、
            # sizeの計算  size = (Input_size - Kernel_size + 2*Padding_size)/ Stride_size + 1

            nn.ReLU(),
            # 画像サイズの変化20*20→9*9
            init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, kernel_size=3, stride=1)),  # 画像サイズの変化9*9→7*7
            nn.ReLU(),
            Flatten(),  # 画像形式を1次元に変換
            init_(nn.Linear(64 * 7 * 7, 512)),  # 64枚の7×7の画像を、512次元のoutputへ
            nn.ReLU()
        )

        # 結合パラメータの初期化関数
        def init_(module): return init(module, gain=1.0)

        # Criticの定義
        self.critic = init_(nn.Linear(512, 1))  # 状態価値なので出力は1つ

        # 結合パラメータの初期化関数
        def init_(module): return init(module, gain=0.01)

        # Actorの定義
        self.actor = init_(nn.Linear(512, n_out))  # 行動を決めるので出力は行動の種類数

        # ネットワークを訓練モードに設定
        self.train()

    def forward(self, x):
        '''ネットワークのフォワード計算を定義します'''
        input = x / 255.0  # 画像のピクセル値0-255を0-1に正規化する
        conv_output = self.conv(input)  # Convolution層の計算
        critic_output = self.critic(conv_output)  # 状態価値の計算
        actor_output = self.actor(conv_output)  # 行動の計算

        return critic_output, actor_output

    def act(self, x):
        '''状態xから行動を確率的に求めます'''
        value, actor_output = self(x)  
        # 190324
        # self(x)の動作について、以下のリンクの最下部のFAQに解説を補足しました。
        # https://github.com/YutaroOgawa/Deep-Reinforcement-Learning-Book
        
        probs = F.softmax(actor_output, dim=1)    # dim=1で行動の種類方向に計算
        action = probs.multinomial(num_samples=1)

        return action

    def get_value(self, x):
        '''状態xから状態価値を求めます'''
        value, actor_output = self(x)
        # 190324
        # self(x)の動作について、以下のリンクの最下部のFAQに解説を補足しました。
        # https://github.com/YutaroOgawa/Deep-Reinforcement-Learning-Book
        
        return value

    def evaluate_actions(self, x, actions):
        '''状態xから状態価値、実際の行動actionsのlog確率とエントロピーを求めます'''
        value, actor_output = self(x)
        # 190324
        # self(x)の動作について、以下のリンクの最下部のFAQに解説を補足しました。
        # https://github.com/YutaroOgawa/Deep-Reinforcement-Learning-Book
        
        log_probs = F.log_softmax(actor_output, dim=1)  # dim=1で行動の種類方向に計算
        action_log_probs = log_probs.gather(1, actions)  # 実際の行動のlog_probsを求める

        probs = F.softmax(actor_output, dim=1)  # dim=1で行動の種類方向に計算
        dist_entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, dist_entropy


# In[9]:


# エージェントが持つ頭脳となるクラスを定義、全エージェントで共有する


class Brain(object):
    def __init__(self, actor_critic):

        self.actor_critic = actor_critic  # actor_criticはクラスNetのディープ・ニューラルネットワーク

        # 結合パラメータをロードする場合
        #filename = 'weight.pth'
        #param = torch.load(filename, map_location='cpu')
        # self.actor_critic.load_state_dict(param)

        # パラメータ更新の勾配法の設定
        self.optimizer = optim.RMSprop(
            actor_critic.parameters(), lr=lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        '''advanced計算した5つのstepの全てを使って更新します'''
        obs_shape = rollouts.observations.size()[2:]  # torch.Size([4, 84, 84])
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, 1))

        # 注意：各変数のサイズ
        # rollouts.observations[:-1].view(-1, *obs_shape) torch.Size([80, 4, 84, 84])
        # rollouts.actions.view(-1, 1) torch.Size([80, 1])
        # values torch.Size([80, 1])
        # action_log_probs torch.Size([80, 1])
        # dist_entropy torch.Size([])

        values = values.view(num_steps, num_processes,
                             1)  # torch.Size([5, 16, 1])
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values  # torch.Size([5, 16, 1])
        value_loss = advantages.pow(2).mean()

        action_gain = (advantages.detach() * action_log_probs).mean()
        # detachしてadvantagesを定数として扱う

        total_loss = (value_loss * value_loss_coef -
                      action_gain - dist_entropy * entropy_coef)

        self.optimizer.zero_grad()  # 勾配をリセット
        total_loss.backward()  # バックプロパゲーションを計算
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        #  一気に結合パラメータが変化しすぎないように、勾配の大きさは最大0.5までにする

        self.optimizer.step()  # 結合パラメータを更新


# In[10]:


# Breakoutを実行する環境のクラス


class Environment:
    def run(self):

        # seedの設定
        seed_num = 1
        torch.manual_seed(seed_num)
        if use_cuda:
            torch.cuda.manual_seed(seed_num)

        # 実行環境を構築
        torch.set_num_threads(seed_num)
        envs = [make_env(ENV_NAME, seed_num, i) for i in range(NUM_PROCESSES)]
        envs = SubprocVecEnv(envs)  # マルチプロセスの実行環境にする

        # 全エージェントが共有して持つ頭脳Brainを生成
        n_out = envs.action_space.n  # 行動の種類は4
        actor_critic = Net(n_out).to(device)  # GPUへ
        global_brain = Brain(actor_critic)

        # 格納用変数の生成
        obs_shape = envs.observation_space.shape  # (1, 84, 84)
        obs_shape = (obs_shape[0] * NUM_STACK_FRAME,
                     *obs_shape[1:])  # (4, 84, 84)
        # torch.Size([16, 4, 84, 84])
        current_obs = torch.zeros(NUM_PROCESSES, *obs_shape).to(device)
        rollouts = RolloutStorage(
            NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape)  # rolloutsのオブジェクト
        episode_rewards = torch.zeros([NUM_PROCESSES, 1])  # 現在の試行の報酬を保持
        final_rewards = torch.zeros([NUM_PROCESSES, 1])  # 最後の試行の報酬和を保持

        # 初期状態の開始
        obs = envs.reset()
        obs = torch.from_numpy(obs).float()  # torch.Size([16, 1, 84, 84])
        current_obs[:, -1:] = obs  # flameの4番目に最新のobsを格納

        # advanced学習用のオブジェクトrolloutsの状態の1つ目に、現在の状態を保存
        rollouts.observations[0].copy_(current_obs)

        # 実行ループ
        for j in tqdm(range(NUM_UPDATES)):
            # advanced学習するstep数ごとに計算
            for step in range(NUM_ADVANCED_STEP):

                # 行動を求める
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])

                cpu_actions = action.squeeze(1).cpu().numpy()  # tensorをNumPyに

                # 1stepの並列実行、なお返り値のobsのsizeは(16, 1, 84, 84)
                obs, reward, done, info = envs.step(cpu_actions)

                # 報酬をtensorに変換し、試行の総報酬に足す
                # sizeが(16,)になっているのを(16, 1)に変換
                reward = np.expand_dims(np.stack(reward), 1)
                reward = torch.from_numpy(reward).float()
                episode_rewards += reward

                # 各実行環境それぞれについて、doneならmaskは0に、継続中ならmaskは1にする
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])

                # 最後の試行の総報酬を更新する
                final_rewards *= masks  # 継続中の場合は1をかけ算してそのまま、done時には0を掛けてリセット
                # 継続中は0を足す、done時にはepisode_rewardsを足す
                final_rewards += (1 - masks) * episode_rewards

                # 試行の総報酬を更新する
                episode_rewards *= masks  # 継続中のmaskは1なのでそのまま、doneの場合は0に

                # masksをGPUへ
                masks = masks.to(device)

                # 現在の状態をdone時には全部0にする
                # maskのサイズをtorch.Size([16, 1])→torch.Size([16, 1, 1 ,1])へ変換して、かけ算
                current_obs *= masks.unsqueeze(2).unsqueeze(2)

                # frameをstackする
                # torch.Size([16, 1, 84, 84])
                obs = torch.from_numpy(obs).float()
                current_obs[:, :-1] = current_obs[:, 1:]  # 0～2番目に1～3番目を上書き
                current_obs[:, -1:] = obs  # 4番目に最新のobsを格納

                # メモリオブジェクトに今stepのtransitionを挿入
                rollouts.insert(current_obs, action.data, reward, masks)

            # advancedのfor loop終了

            # advancedした最終stepの状態から予想する状態価値を計算
            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.observations[-1]).detach()

            # 全stepの割引報酬和を計算して、rolloutsの変数returnsを更新
            rollouts.compute_returns(next_value)

            # ネットワークとrolloutの更新
            global_brain.update(rollouts)
            rollouts.after_update()

            # ログ：途中経過の出力
            if j % 100 == 0:
                print("finished frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".
                      format(j*NUM_PROCESSES*NUM_ADVANCED_STEP,
                             final_rewards.mean(),
                             final_rewards.median(),
                             final_rewards.min(),
                             final_rewards.max()))

            # 結合パラメータの保存
            if j % 12500 == 0:
                torch.save(global_brain.actor_critic.state_dict(),
                           'weight_'+str(j)+'.pth')
        
        # 実行ループの終了
        torch.save(global_brain.actor_critic.state_dict(), 'weight_end.pth')
        


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # torch.cuda.set_device(1)
    # launch()
    device = "cuda"
    # 実行
    breakout_env = Environment()
    breakout_env.run()
    # plt.show()


# In[ ]:




