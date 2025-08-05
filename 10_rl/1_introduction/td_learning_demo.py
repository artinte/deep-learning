import numpy as np

# Define a 5x5 grid world
# 0 = normal state, 100 = goal state (reward), -100 = trap (penalty)
grid = np.zeros((5, 5))
grid[4, 4] = 100  # Goal at bottom-right corner
grid[2, 2] = -100  # Trap in the middle

# Initialize all state values to 0
state_values = np.zeros((5, 5))

# Define possible actions: up, right, down, left
actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# TD(0) parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
episodes = 1000  # Number of training episodes

def get_reward(state):
    """Get the immediate reward for a state"""
    x, y = state
    return grid[x, y]

def is_terminal(state):
    """Check if a state is terminal (goal or trap)"""
    x, y = state
    return grid[x, y] != 0  # Terminal if not normal state

def take_action(state, action):
    """Execute an action and return the new state"""
    x, y = state
    dx, dy = action
    # Ensure we stay within grid boundaries
    new_x = max(0, min(x + dx, 4))
    new_y = max(0, min(y + dy, 4))
    return (new_x, new_y)

# Train using TD(0) algorithm
for _ in range(episodes):
    # Randomly select starting position (not a terminal state)
    while True:
        x = np.random.randint(5)
        y = np.random.randint(5)
        if not is_terminal((x, y)):
            current_state = (x, y)
            break
    
    # Run through the episode
    while not is_terminal(current_state):
        # Randomly choose an action
        action = actions[np.random.randint(4)]
        
        # Take action to get next state
        next_state = take_action(current_state, action)
        
        # Get reward for the transition
        reward = get_reward(next_state)
        
        # TD(0) update rule: V(S_t) ← V(S_t) + α[reward + γV(S_{t+1}) - V(S_t)]
        current_value = state_values[current_state]
        next_value = state_values[next_state]
        
        # Apply the update
        state_values[current_state] = current_value + alpha * (reward + gamma * next_value - current_value)
        
        # Move to next state
        current_state = next_state

# Print the learned state values (rounded to 1 decimal place)
print("Learned state values:")
print(np.round(state_values, 1))
