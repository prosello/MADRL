from madrl_environments import AbstractMAEnv, Agent
from rltools.util import EzPickle
from gym import spaces
import numpy as np

class TrackerAgent(Agent):
    def __init__(self, det_dim=4, pred_dim=6):
        self.det_dim = det_dim
        self.pred_dim = pred_dim
        self.obs_dim = det_dim + pred_dim

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim))

    @property
    def action_space(self):
        return spaces.Discrete(5)

class MotKittiEnv(AbstractMAEnv, EzPickle):
    def __init__(self, n_trackers=10, det_dim=4, pred_dim=6):
        EzPickle.__init__(self, n_trackers, det_dim, pred_dim)
        self.n_trackers = n_trackers
        self.det_dim = det_dim
        self.pred_dim = pred_dim
        self.setup()

    def get_param_values(self):
        return self.__dict__

    def setup(self):
        self.trackers = [TrackerAgent(det_dim=self.det_dim, pred_dim=self.pred_dim) for t in range(self.n_trackers)]

    @property
    def agents(self):
        return self.trackers

    @property
    def reward_mech(self):
        return "global" # TODO: What to set this too?

    def reset(self):
        init_obs = np.array([np.random.rand(10) for i in range(self.n_trackers)])
        return init_obs

    def step(self, actions):
        obs = np.array([np.random.rand(10) for i in range(self.n_trackers)])

        global_reward = np.mean(obs)
        rewards = [global_reward for i in range(self.n_trackers)] # all receive same reward?
        done = False
        return obs, rewards, done, {}

    def render(self):
        pass

if __name__ == "__main__":
    n_trackers = 3
    env = MotKittiEnv(n_trackers=n_trackers)
    env.reset()
    for i in range(10):
        env.render()
        actions = np.array([agent.action_space.sample() for agent in env.agents])
        obs, rewards, done, _ = env.step(actions)
        print("\nStep:", i)
        print("Actions:", actions)
        print("Obs:", obs)
        print("Rewards:", rewards)
        if done:
            break