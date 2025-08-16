import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from cartpole import CartPole


class PolicyNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_reinforce(episodes=2000, gamma=0.99, lr=1e-2):
    env = CartPole()
    policy = PolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    for ep in range(episodes):
        states, actions, rewards = [], [], []
        env.reset()
        env.set_random_IC(mult_x=100, mult_theta=np.pi/18)
        state = env.get_state()
        done = False

        while not done:
            state_tensor = torch.tensor([state], dtype=torch.float32)
            logits = policy(state_tensor)
            probs = torch.softmax(logits, dim=-1).detach().numpy()[0]

            action = np.random.choice(len(probs), p=probs) - 1  # map {0,1,2} -> {-1,0,1}
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action + 1)  # shift back to {0,1,2} for CE loss
            rewards.append(reward)

            state = next_state

        # Compute returns
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = (np.array(returns) - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Train
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        logits = policy(states_tensor)
        losses = loss_fn(logits, actions_tensor)
        loss = torch.mean(losses * returns_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}, total reward = {sum(rewards)}")

    torch.save(policy.state_dict(), "reinforce_cartpole_custom.pt")
    return policy


if __name__ == "__main__":
    trained_policy = train_reinforce()
