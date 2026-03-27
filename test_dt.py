import gymnasium as gym
import d3rlpy
import numpy as np
from d3rlpy.algos import DiscreteDecisionTransformerConfig, StatefulTransformerWrapper, GreedyTransformerActionSampler
from gymnasium.wrappers import RecordVideo

env = gym.make('LunarLander-v3', render_mode='rgb_array')
env = RecordVideo(env, video_folder='./videos', name_prefix='dt_test', disable_logger=True)
model = DiscreteDecisionTransformerConfig(context_size=20, max_timestep=1000).create(device='cpu')
model.build_with_env(env)
model.load_model('dt_expert.d3')
wrapper = StatefulTransformerWrapper(model, target_return=200.0, action_sampler=GreedyTransformerActionSampler())

obs, info = env.reset()
wrapper.reset()

for i in range(10):
    action = wrapper.predict(obs, 0.0 if i == 0 else reward)
    if isinstance(action, np.ndarray):
        action = int(action[0])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}: Action={action}, reward={reward}, term={terminated}, trunc={truncated}")
    off_screen = abs(obs[0]) > 1.1 or obs[1] > 1.2 or obs[1] < -0.1
    if terminated or truncated or off_screen:
        print(f"DONE at step {i}! off_screen={off_screen}, x={obs[0]}, y={obs[1]}")
        break

env.close()
