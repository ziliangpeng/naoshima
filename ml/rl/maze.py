import numpy as np

# Constants
N_STATES = 6  # number of states
ACTIONS = ["left", "right"]  # available actions
EPSILON = 0.9  # for exploitation
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODES = 42  # maximum episodes
FRESH_TIME = 0.3  # fresh time for one move

# Create Q-table
q_table = np.zeros((N_STATES, len(ACTIONS)))


def choose_action(state, q_table):
    state_actions = q_table[state, :]
    if np.random.uniform() > EPSILON or np.all(state_actions == 0):
        action_index = np.random.randint(0, 2)
    else:
        action_index = np.argmax(state_actions)
    return action_index


def get_env_feedback(state, action_index):
    if action_index == 0:  # move left
        if state == 0:
            next_state = state  # reach the wall
            reward = 0
        elif state == 2:
            next_state = state - 1
            # reward = 1
            reward = 0
        else:
            next_state = state - 1
            reward = 0
    else:  # move right
        if state == N_STATES - 1:
            next_state = "terminal"
            reward = 1
        else:
            next_state = state + 1
            # reward = 0
            reward = 1
    return next_state, reward


def update_env(state, episode, step_counter):
    env_list = ["-"] * (N_STATES - 1) + ["T"]  # '---------T' our environment
    if state == "terminal":
        print("Episode {}: total_steps = {}".format(episode + 1, step_counter))
    else:
        env_list[state] = "o"
        # print(''.join(env_list))


def rl():
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0
        is_terminated = False
        update_env(state, episode, step_counter)
        while not is_terminated:
            action_index = choose_action(state, q_table)
            next_state, reward = get_env_feedback(state, action_index)
            q_predict = q_table[state, action_index]
            if next_state != "terminal":
                q_target = reward + GAMMA * np.max(q_table[next_state, :])
            else:
                q_target = reward
                is_terminated = True
            q_table[state, action_index] += ALPHA * (q_target - q_predict)
            state = next_state
            update_env(state, episode, step_counter + 1)
            step_counter += 1


if __name__ == "__main__":
    rl()
