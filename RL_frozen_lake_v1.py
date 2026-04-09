import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)

print("Action Space:", env.action_space)
print("Observation Space:", env.observation_space)

state_size = env.observation_space.n
action_size = env.action_space.n

print("State size:", state_size)
print("Action size:", action_size)

Q = np.zeros((state_size, action_size))
# print("Initial Q-table:", Q)

# for _ in range(50):
#     print("Random number:", np.random.rand())
#     if np.random.rand() < 1.0:
#         print("--------Exploring: Taking random action")
#     else:        print("Exploiting: Taking action with max Q-value")

episodes = 1000
learning_rate = 0.8
gamma = 0.95
epsilon = 1.0

# Actions:
# 0 = LEFT
# 1 = DOWN
# 2 = RIGHT
# 3 = UP

# 0   1   2   3
# 4   5   6   7
# 8   9  10  11
# 12 13  14  15

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    # print(f"Episode {episode } starting... Initial state: {state}")
    while not done:
        # explore vs exploit
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, done, truncated, _ = env.step(action)

        # print(f"Episode {episode}, State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")

        # Q-learning update
        Q[state][action] = Q[state][action] + learning_rate * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action]
        )

        state = next_state

    epsilon *= 0.995  # reduce randomness

print("Learned Q-table:")
print(Q)
'''

Q = [
 [0.73508244 , 0.77378094, 0.69833722, 0.73509187],
 [0.73509189 , 0.,         0.,         0.55626252],
 [0.55626252 , 0.,         0.,         0.        ],
 [0.         , 0.,         0.,         0.        ],
 [0.77378054 , 0.81450625, 0.,         0.73485666],
 [0.         , 0.,         0.,         0.        ],
 [0.         , 0.,         0.,         0.        ],
 [0.         , 0.,         0.,         0.        ],
 [0.81450416 , 0.,         0.857375,   0.77378053],
 [0.81449582 , 0.7297976,  0.9025,     0.        ],
 [0.85737281 , 0.95,       0.,         0.        ],
 [0.         , 0.,         0.,         0.        ],
 [0.         , 0.,         0.,         0.        ],
 [0.         , 0.,         0.,         0.850516  ],
 [0.         , 0.9424,     1.,         0.90249953],
 [0.         , 0.,         0.,         0.        ],]
# env.render()
# print(env.unwrapped.desc)
print("Q-table:", Q)
policy = np.argmax(Q, axis=1)


action_map = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑"
}

grid = policy.reshape(4, 4)
visual = np.empty((4, 4), dtype=object)

desc = env.unwrapped.desc


for i in range(4):
    for j in range(4):
        if desc[i][j] == b'H':
            visual[i][j] = 'X'
        elif desc[i][j] == b'G':
            visual[i][j] = 'G'
        else:
            visual[i][j] = action_map[grid[i][j]]

print("Policy Grid:")
print(visual)
'''