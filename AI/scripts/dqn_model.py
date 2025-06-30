import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    """
    Deep Q-Network for maze navigation using both global and local wall state.
    """
    def __init__(self, global_state_size, local_state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(global_state_size + local_state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, global_state, local_state):
        x = torch.cat([global_state, local_state], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, global_state_size, local_state_size, action_size, device='cpu'):
        self.global_state_size = global_state_size
        self.local_state_size = local_state_size
        self.action_size = action_size
        self.device = device
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory_size = 10000
        self.batch_size = 64
        self.update_target_every = 1000
        self.q_network = DQN(global_state_size, local_state_size, action_size).to(device)
        self.target_network = DQN(global_state_size, local_state_size, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = []
        self.step_count = 0
        self._update_target_network()

    def _update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, global_state, local_state, action, reward, next_global_state, next_local_state, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((global_state, local_state, action, reward, next_global_state, next_local_state, done))

    def act(self, global_state, local_state, training=True):
        if training and np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        global_state = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        local_state = torch.FloatTensor(local_state).unsqueeze(0).to(self.device)
        q_values = self.q_network(global_state, local_state)
        return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        global_states = torch.FloatTensor(np.array([self.memory[i][0] for i in batch])).to(self.device)
        local_states = torch.FloatTensor(np.array([self.memory[i][1] for i in batch])).to(self.device)
        actions = torch.LongTensor([self.memory[i][2] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([self.memory[i][3] for i in batch]).to(self.device)
        next_global_states = torch.FloatTensor(np.array([self.memory[i][4] for i in batch])).to(self.device)
        next_local_states = torch.FloatTensor(np.array([self.memory[i][5] for i in batch])).to(self.device)
        dones = torch.BoolTensor([self.memory[i][6] for i in batch]).to(self.device)
        current_q_values = self.q_network(global_states, local_states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_global_states, next_local_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self._update_target_network()
        return loss.item()

    def save(self, filename):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count'] 