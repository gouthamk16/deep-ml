import numpy as np

def expected_vs_sample_updates(
    P: np.ndarray, 
    R: np.ndarray, 
    gamma: float, 
    alpha: float, 
    num_sweeps: int, 
    seed: int = 42) -> tuple:
    """
    Compare expected and sample updates for Q-value estimation.
    
    Args:
        P: np.ndarray of shape (num_states, num_actions, num_states), transition probabilities
        R: np.ndarray of shape (num_states, num_actions), rewards
        gamma: float, discount factor
        alpha: float, learning rate for sample updates
        num_sweeps: int, number of sweeps through all state-action pairs
        seed: int, random seed for reproducibility
    
    Returns:
        tuple: (Q_expected, Q_sample) both np.ndarray of shape (num_states, num_actions)
    """
    np.random.seed(seed)

    num_states, num_actions, _ = P.shape

    # Initialize Q tables
    Q_expected, Q_sample = np.zeros((num_states, num_actions)), np.zeros((num_states, num_actions))
    
    # Iterate over the sweeps
    for _ in range(num_sweeps):
        old_Q = Q_expected.copy() # All updates in this sweep should use the q values from the start of the sweep
        for state in range(num_states):
            for action in range(num_actions):
                expected_future = 0
                # Loop over every possible next state
                for next_state in range(num_states):
                    probability = P[state, action, next_state]
                    # Best action value
                    best_next_value = np.max(old_Q[next_state])
                    expected_future += probability * best_next_value
                Q_expected[state, action] = R[state, action] + (gamma * expected_future)
        
        for state in range(num_states):
            for action in range(num_actions):
                # Randomly chose a next stage
                next_state = np.random.choice(num_states, p=P[state, action])
                current_estimate = Q_sample[state, action]
                best_next_value = np.max(Q_sample[next_state])
                reward = R[state, action]
                target = reward + (gamma * best_next_value)
                error = target - current_estimate
                Q_sample[state, action] = current_estimate + (alpha * error)



    return (np.round(Q_expected, 4), np.round(Q_sample, 4))