import gymnasium as gym

env = gym.make("CartPole-v1")

print("Action Space:", env.action_space)
print("Observation Space:", env.observation_space)

state, _ = env.reset()
print(state)
for _ in range(10):
    action = env.action_space.sample()  # random action
    print("action:", action)
    next_state, reward, done, truncated, _ = env.step(action)

    print("State:", next_state, "Reward:", reward)

    if done or truncated:
        state, _ = env.reset()
    else:
        state = next_state

env.close()