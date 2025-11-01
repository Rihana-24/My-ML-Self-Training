import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym
import matplotlib.pyplot as plt

# Set up the environment
env = gym.make('CartPole-v1')
state_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

# Parameters
learning_rate = 0.01
num_episodes = 500
gamma = 0.99
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Define the neural network model (policy network)
def build_policy_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(state_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_shape, activation='softmax')  # softmax = probabilities
    ])
    return model

policy_model = build_policy_model()

# Function to select an action based on policy
def choose_action(state):
    state = tf.expand_dims(state, axis=0)
    action_probs = policy_model(state).numpy().squeeze()
    action = np.random.choice(action_shape, p=action_probs)
    return action

# Discount rewards
def discount_rewards(rewards, gamma=0.99):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0
    for t in reversed(range(len(rewards))):
        cumulative = cumulative * gamma + rewards[t]
        discounted[t] = cumulative
    # Normalization
    discounted = (discounted - np.mean(discounted)) / (np.std(discounted) + 1e-8)
    return discounted

# Training function
def train_on_episode(states, actions, rewards):
    discounted_rewards = discount_rewards(rewards, gamma)
    with tf.GradientTape() as tape:
        action_probs = policy_model(tf.convert_to_tensor(states, dtype=tf.float32))
        indices = tf.range(len(actions))
        chosen_action_probs = tf.gather_nd(action_probs, tf.stack([indices, actions], axis=1))
        log_probs = tf.math.log(chosen_action_probs + 1e-8)
        loss = -tf.reduce_mean(log_probs * discounted_rewards)
    gradients = tape.gradient(loss, policy_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy_model.trainable_variables))
    return loss.numpy()

# Main training loop
all_rewards = []
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    episode_states, episode_actions, episode_rewards = [], [], []

    while not done:
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)

        state = next_state

    loss = train_on_episode(np.array(episode_states),
                            np.array(episode_actions),
                            np.array(episode_rewards))

    total_reward = sum(episode_rewards)
    all_rewards.append(total_reward)

    print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward} | Loss: {loss:.4f}")

# Plot learning curve
plt.plot(all_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Policy Gradient Training on CartPole-v1")
plt.show()
