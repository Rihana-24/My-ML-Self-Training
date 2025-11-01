# Import packages  
import numpy as np
import random 

# Define the environment (4x4 grid)
num_states = 16  # 4x4 grid
num_actions = 4  # (0=Up, 1=Right, 2=Down, 3=Left)
q_table = np.zeros((num_states, num_actions))  # initialize the Q-table with zeros

# Define parameters 
alpha = 0.1   # learning rate
gamma = 0.9   # discount factor
epsilon = 0.1 # exploration rate
num_episodes = 1000  # number of episodes

# Define a simple reward structure
rewards = np.zeros(num_states)  # initialize the reward vector with zeros
rewards[15] = 1  # Goal state with a reward

# Function to determine the next state based on the action
def get_next_state(state, action):
    if action == 0 and state >= 4:               # Up
        next_state = state - 4
    elif action == 1 and (state + 1) % 4 != 0:   # Right
        next_state = state + 1
    elif action == 2 and state < 12:             # Down
        next_state = state + 4
    elif action == 3 and state % 4 != 0:         # Left
        next_state = state - 1
    else:
        next_state = state  # stay in same state if invalid move
    return next_state

# Q-learning algorithm
for episode in range(num_episodes):
    state = random.randint(0, 15)  # Random initial state
    while state != 15:  # Continue until goal is reached
        # Choose action (epsilon-greedy)
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit best action
        
        # Take action
        next_state = get_next_state(state, action)
        reward = rewards[next_state]

        # Q-learning update rule
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state, action] = new_value

        # Move to next state
        state = next_state

# Display the learned Q-table
print("Q-table after training:")
print(q_table)
