import numpy as np
import torch
from cartpole_physics import CartPole
from reinforce import PolicyNet
import argparse

parser = argparse.ArgumentParser()
# model parameters
parser.add_argument("--hidden_dim", default=64, type=int, help="Size of the hidden layer.")
# other
parser.add_argument("--model_name", default="gym_cartpole_model.keras", type=str, help="Output model path.")
parser.add_argument("--print_when_eval", default=True, type=bool, help="Print debug info during training.")
parser.add_argument("--episodes_eval", default=5, type=int, help="Number of epochs for evaluation.")
parser.add_argument("--save_epochs", default=True, type=bool, help="Save each epoch to a csv file.")


def evaluate(env, policy, episodes=5, print_when_eval=True, save_epochs=True):
    policy.eval()

    reward_per_episode = []
    for ep in range(episodes):
        data = []
        env.reset()
        env.set_random_IC()
        state = env.get_state()
        state = torch.from_numpy(state).float()

        done = False
        total_reward = 0
        steps = 0
        while not done:
            #state_tensor = torch.tensor([state], dtype=torch.float32)
            state_tensor = state.unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_tensor)
                action = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()

            # map back {0,1,2} → {-1,0,1}
            action = action - 1
            data.append(list(state))

            state, reward, done = env.step_wrapped(action)
            state = torch.from_numpy(state).float()

            total_reward += reward
            steps += 1

        reward_per_episode.append(total_reward)
        if print_when_eval:
            print(f"Episode {ep+1}: survived {steps} steps, reward = {total_reward}")

        if save_epochs:
            np.savetxt(f'data/episode_{ep+1}.csv', data, delimiter=',')

    reward_per_episode = np.array(reward_per_episode).astype(int)
    return reward_per_episode


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    env = CartPole()
    policy = PolicyNet(args.hidden_dim)
    policy.load_state_dict(torch.load(args.model_name, map_location="cpu"))
    evaluate(env, policy, args.episodes_eval, args.print_when_eval, args.save_epochs)
