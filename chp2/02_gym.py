import gymnasium as gym

env = gym.make("CartPole-v1", render_mode='human')
env.reset()

total_reward = 0.0
total_steps = 0


while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    total_steps += 1
    if terminated:
        break

print(f"Episode done in {total_steps} steps, total reward {total_reward:.2f}")