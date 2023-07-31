import numpy as np

# Constants
N_STATES = 25  # number of states
ACTIONS = ['up', 'down', 'left', 'right']  # available actions
EPSILON = 0.9  # for exploitation
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODES = 9  # maximum episodes
FRESH_TIME = 0.3  # fresh time for one move

# Create Q-table
q_table = np.zeros((N_STATES, len(ACTIONS)))


def choose_action(state, q_table):
    state_actions = q_table[state, :]
    if np.random.uniform() > EPSILON or np.all(state_actions == 0):
        action_index = np.random.randint(0, len(ACTIONS))
    else:
        action_index = np.argmax(state_actions)
    return action_index


def get_env_feedback(state, action_index, maze):
    row, col = state // 5, state % 5
    if ACTIONS[action_index] == 'up':
        next_row, next_col = row - 1, col
    elif ACTIONS[action_index] == 'down':
        next_row, next_col = row + 1, col
    elif ACTIONS[action_index] == 'left':
        next_row, next_col = row, col - 1
    elif ACTIONS[action_index] == 'right':
        next_row, next_col = row, col + 1

    if next_row < 0 or next_row >= 5 or next_col < 0 or next_col >= 5 or maze[next_row][next_col] == 'X':
        next_state = state
        reward = -1
    elif maze[next_row][next_col] == 'G':
        next_state = 24
        reward = 1
    else:
        next_state = next_row * 5 + next_col
        # Calculate the distance to the goal state
        current_distance = abs(row - 4) + abs(col - 4)
        next_distance = abs(next_row - 4) + abs(next_col - 4)
        # Give a reward of 1 if the agent moves closer to the goal, and -1 if it moves farther away
        if next_distance < current_distance:
            reward = 1
        else:
            reward = -1

    return next_state, reward


def update_env(state, episode, step_counter, maze):
    env_list = []
    for i in range(5):
        for j in range(5):
            if maze[i][j] == 'X':
                env_list.append('X')
            elif maze[i][j] == 'G':
                env_list.append('G')
            elif i * 5 + j == state:
                env_list.append('A')
            else:
                env_list.append('-')
        env_list.append('\n')
    print(''.join(env_list))

    if state == 24:
        print('Episode {}: total_steps = {}'.format(episode + 1, step_counter))


# Define the maze
maze = [
    ['O', 'O', 'O', 'O', 'O'],
    ['O', 'X', 'O', 'X', 'O'],
    ['O', 'O', 'O', 'O', 'O'],
    ['O', 'X', 'O', 'X', 'O'],
    ['O', 'O', 'O', 'O', 'G']
]

# Main loop
for episode in range(MAX_EPISODES):
    state = 0
    step_counter = 0
    is_terminated = False
    update_env(state, episode, step_counter, maze)

    while not is_terminated:
        action_index = choose_action(state, q_table)
        next_state, reward = get_env_feedback(state, action_index, maze)
        q_predict = q_table[state, action_index]
        if next_state != 24:
            q_target = reward + GAMMA * np.max(q_table[next_state, :])
        else:
            q_target = reward
            is_terminated = True

        q_table[state, action_index] += ALPHA * (q_target - q_predict)
        state = next_state
        step_counter += 1
        update_env(state, episode, step_counter, maze)