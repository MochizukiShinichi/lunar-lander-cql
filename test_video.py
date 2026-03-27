import gymnasium as gym
import d3rlpy
from d3rlpy.algos import DiscreteDecisionTransformerConfig, StatefulTransformerWrapper, GreedyTransformerActionSampler
from gymnasium.wrappers import RecordVideo

env = gym.make('LunarLander-v3', render_mode='rgb_array')
env = RecordVideo(env, video_folder='./videos', name_prefix='dt_expert', disable_logger=True)
model = DiscreteDecisionTransformerConfig(context_size=20, max_timestep=1000).create(device='cpu')
model.build_with_env(env)
model.load_model('dt_expert.d3')
wrapper = StatefulTransformerWrapper(model, target_return=200.0, action_sampler=GreedyTransformerActionSampler())

obs, info = env.reset()
wrapper.reset()
done = False
step_count = 0
reward = 0.0

while not done:
    action = wrapper.predict(obs, reward)
    obs, reward, terminated, truncated, info = env.step(action)
    off_screen = abs(obs[0]) > 1.1 or obs[1] > 1.2 or obs[1] < -0.1
    done = terminated or truncated or off_screen or step_count > 400
    step_count += 1
print(f'Steps taken: {step_count}')
env.close()
