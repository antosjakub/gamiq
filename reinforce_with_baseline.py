import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from cartpole_physics import CartPole
import argparse

parser = argparse.ArgumentParser()
# model parameters
parser.add_argument("--hidden_dim", default=8, type=int, help="Size of the hidden layer.")
parser.add_argument("--episodes", default=10000, type=int, help="Number of epochs.")
parser.add_argument("--gamma", default=0.99, type=float, help="Gamma.")
parser.add_argument("--learning_rate_initial", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--learning_rate_final", default=0.005, type=float, help="Learning rate.")
# other
parser.add_argument("--save_model", default=False, action="store_true", help="Wheather or not to same the model to disk.")
parser.add_argument("--model_name", default="gym_cartpole_model.pt", type=str, help="Output model path.")
parser.add_argument("--print_when_training", default=False, action="store_true", help="Print debug info during training.")
parser.add_argument("--evaluate", default=False, action="store_true", help="Run evaluation after training.")
parser.add_argument("--episodes_eval", default=5, type=int, help="Number of epochs for evaluation.")
parser.add_argument("--print_when_eval", default=False, action="store_true", help="Print debug info during training.")
parser.add_argument("--save_eval_episodes", default=False, action="store_true", help="Save each epoch to a csv file.")
# python reinforce_with_baseline.py --print_when_train --evaluate --print_when_eval --hidden_dim=16 --episodes=3000 --gamma=0.99 --learning_rate_initial=0.005 --learning_rate_final=0.0001 --save_eval_episodes


class PolicyNet(nn.Module):
    def __init__(self, hidden_dim, input_dim=4, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class ValueNet(nn.Module):
    def __init__(self, hidden_dim, input_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_reinforce(env, policy, value, learning_rate_initial, learning_rate_final, episodes, gamma, print_when_training):
    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=learning_rate_initial)
    optimizer_value = torch.optim.Adam(value.parameters(), lr=learning_rate_initial)
    loss_policy_ce = nn.CrossEntropyLoss(reduction="none")
    loss_value_mse = nn.MSELoss()

    decay_steps = episodes # 1 update per episode
    scheduler_policy = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_policy,
        T_max=decay_steps,
        eta_min=learning_rate_final
    )
    scheduler_value = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_value,
        T_max=decay_steps,
        eta_min=learning_rate_final
    )


    #easiest_IC = np.array([0,0,np.pi/90,np.pi/90])
    #hardest_IC = np.array([200,50,np.pi/18,np.pi/18])
    #IC = easiest_IC
    #IC = hardest_IC / 5
    past_rewards = []
    for ep in range(episodes):
        states, actions, rewards = [], [], []

        env.reset()
        #if ep % 500 == 0:
        #    fraction = ep/episodes
        #    IC = easiest_IC*(1-fraction) + hardest_IC*fraction
        #    #print(ep, fraction, IC)
        #    if ep != 0: print("Increasing diffuculty of initial conditions!")
        #env.set_random_IC(*IC)
        env.set_random_IC()
        state = env.get_state()
        state = torch.from_numpy(state).float()
        done = False

        while not done:
            #state_tensor = torch.tensor([state], dtype=torch.float32)
            state_tensor = state.unsqueeze(0)

            logits = policy(state_tensor)
            probs = torch.softmax(logits, dim=-1).detach().numpy()[0]

            action = np.random.choice(len(probs), p=probs) - 1  # map {0,1,2} -> {-1,0,1}
            next_state, reward, done = env.step_wrapped(action)
            next_state = torch.from_numpy(next_state).float()

            states.append(state)
            actions.append(action + 1)  # shift back to {0,1,2} for CE loss
            rewards.append(reward)

            state = next_state

        # Compute returns
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        #returns = (np.array(returns) - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Train
        #print(type(states), type(states[0]))
        #states_tensor = torch.tensor(states, dtype=torch.float32)
        states_tensor = torch.stack(states)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        # ---- Baseline (value function) ----
        values = value(states_tensor)
        advatages = returns_tensor - values.detach()

        # ---- Actor loss (REINFORCE with baseline) ----
        logits = policy(states_tensor)
        log_probs = loss_policy_ce(logits, actions_tensor)
        actor_loss = torch.mean(log_probs * advatages)

        # ---- Critic loss (MSE between V(s) and return) ----
        critic_loss = loss_value_mse(values, returns_tensor)

        # ---- Optimize ---- (actor)
        optimizer_policy.zero_grad()
        actor_loss.backward()
        optimizer_policy.step()
        # ---- Optimize ---- (critic)
        optimizer_value.zero_grad()
        critic_loss.backward()
        optimizer_value.step()
        # ---- Cosine decay
        scheduler_policy.step()
        scheduler_value.step()

        past_rewards.append(sum(rewards))
        if print_when_training:
            if (ep + 1) % 100 == 0:
                print(f"Episode {ep+1}, reward = {past_rewards[-1]}, average reward = {sum(past_rewards)/100}")
                past_rewards = []

    return policy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    #print("save?", args.save_model)
    #print("save_eval_episodes?", args.save_eval_episodes)
    #print("evaluate?", args.evaluate)
    #print("print train?", args.print_when_training)
    #print("print eval?", args.print_when_eval)

    env = CartPole()
    policy = PolicyNet(args.hidden_dim)
    value = ValueNet(args.hidden_dim)
    trained_policy = train_reinforce(env, policy, value, args.learning_rate_initial, args.learning_rate_final, args.episodes, args.gamma, args.print_when_training)

    if args.save_model:
        torch.save(trained_policy.state_dict(), args.model_name)

    if args.evaluate:
        import test_model
        reward_per_episode = test_model.evaluate(env, policy, args.episodes_eval, args.print_when_eval, save_epochs=args.save_eval_episodes)
        print(f"average reward: {np.sum(reward_per_episode)/args.episodes_eval}, rewards = {reward_per_episode}")


