import numpy as np
import gym
import matplotlib.pyplot as plt

num_episodes = 5_000
max_number_of_step = 200
num_consecutive_iterations = 100
last_time_steps = np.zeros(num_consecutive_iterations)
goal_average_steps = 195

env = gym.make('CartPole-v1')
observation = env.reset()

q_table = np.random.uniform(low=-1, high=1, size=(4 ** 4, env.action_space.n))

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num+1)[1:-1]


def digitize_state(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation

    digitized = [np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)),
                 np.digitize(cart_v, bins=bins(-3.0, 3.0, 4)),
                 np.digitize(pole_angle, bins=bins(-0.5, 0.5, 4)),
                 np.digitize(pole_v, bins=bins(-2.0, 2.0, 4))]
    
    return sum([x * (4 ** i) for i, x in enumerate(digitized)])


def get_action(state, action, observation, reward):
    next_state = digitize_state(observation)
    next_action = np.argmax(q_table[next_state])

    alpha = 0.2
    gamma = 0.99

    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
        alpha * (reward + gamma * q_table[next_state, next_action])
    
    return next_action, next_state

step_list = []
for episode in range(num_episodes):
    observation = env.reset()
    state = digitize_state(observation)
    action = np.argmax(q_table[state])
    episode_reward = 0
    for t in range(max_number_of_step):
        env.render()
        observation, reward, done, info = env.step(action)
        action,state = get_action(state, action, observation, reward)
        episode_reward += reward

        if done:
            print(f"{episode} Episode finished after {t+1} time steps mean {last_time_steps.mean()}")
            last_time_steps = np.hstack((last_time_steps[1:], [episode_reward]))
            step_list.append(last_time_steps.mean())
            break
    
    if (last_time_steps.mean() >= goal_average_steps):
        print(f'Episode {episode} train agent successfully!')
        break

plt.plot(step_list)
plt.xlabel('episode')
plt.ylabel('mean_step')
plt.show()