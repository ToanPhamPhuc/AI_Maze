import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from environment import MazeEnvironment
from dqn_model import DQNAgent
import torch

def train_agent(episodes=1000, render_every=100, save_every=100):
    """
    Train the DQN agent on the maze environment.
    
    Args:
        episodes: Number of training episodes
        render_every: Render every N episodes
        save_every: Save model every N episodes
    """
    
    # Create environment and agent
    env = MazeEnvironment(height=8, width=8, cell_size=30)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env.state_size, env.action_space, device)
    
    # Training statistics
    scores = []
    losses = []
    epsilons = []
    
    print("Training on device: {}".format(device))
    print(f"Environment: {env.width}x{env.height} maze")
    print(f"State size: {env.state_size}, Action size: {env.action_space}")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_losses = []
        
        # Render occasionally
        if episode % render_every == 0:
            screen = env.render()
        
        while True:
            # Choose action
            action = agent.act(state, training=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render occasionally
            if episode % render_every == 0:
                env.render(screen)
            
            if done:
                break
        
        # Record statistics
        scores.append(total_reward)
        losses.append(np.mean(episode_losses) if episode_losses else 0)
        epsilons.append(agent.epsilon)
        
        # Print progress
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:])
            avg_loss = np.mean(losses[-10:])
            print(f"Episode {episode}/{episodes} - Score: {total_reward:.1f}, "
                  f"Steps: {steps}, Avg Score: {avg_score:.1f}, "
                  f"Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.3f}")
        
        # Save model periodically
        if episode % save_every == 0 and episode > 0:
            agent.save(f"models/maze_dqn_episode_{episode}.pth")
            print(f"Model saved at episode {episode}")
    
    # Save final model
    agent.save("models/maze_dqn_final.pth")
    print("Training completed! Final model saved.")
    
    # Plot training results
    plot_training_results(scores, losses, epsilons)
    
    return agent, scores, losses

def plot_training_results(scores, losses, epsilons):
    """Plot training statistics."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot scores
    ax1.plot(scores)
    ax1.set_title('Training Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot losses
    ax2.plot(losses)
    ax2.set_title('Training Losses')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    # Plot epsilon
    ax3.plot(epsilons)
    ax3.set_title('Exploration Rate (Epsilon)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

def test_agent(model_path, episodes=10):
    """
    Test a trained agent.
    
    Args:
        model_path: Path to the trained model
        episodes: Number of test episodes
    """
    env = MazeEnvironment(height=8, width=8, cell_size=30)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env.state_size, env.action_space, device)
    
    # Load trained model
    agent.load(model_path)
    agent.epsilon = 0  # No exploration during testing
    
    print(f"Testing agent from {model_path}")
    
    test_scores = []
    test_steps = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        test_scores.append(total_reward)
        test_steps.append(steps)
        print(f"Test Episode {episode + 1}: Score = {total_reward}, Steps = {steps}")
    
    print(f"\nTest Results:")
    print(f"Average Score: {np.mean(test_scores):.2f}")
    print(f"Average Steps: {np.mean(test_steps):.2f}")
    print(f"Success Rate: {np.sum(np.array(test_scores) > 0) / episodes * 100:.1f}%")

if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train the agent
    print("Starting DQN training...")
    agent, scores, losses = train_agent(episodes=500, render_every=50, save_every=100)
    
    # Test the trained agent
    print("\nTesting trained agent...")
    test_agent("models/maze_dqn_final.pth", episodes=10) 