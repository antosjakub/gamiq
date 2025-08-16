import numpy as np
import torch
from cartpole import CartPole   # your custom physics module
from reinforce import PolicyNet     # import the same policy architecture


def evaluate(model_path="reinforce_cartpole_custom.pt", episodes=5):
    env = CartPole()
    policy = PolicyNet()
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()

    for ep in range(episodes):
        data = []
        # set IC
        env.reset()
        #env.set_random_IC(mult_x=500, mult_theta=np.pi/6)
        env.set_random_IC(mult_x=100, mult_theta=np.pi/18)
        state = env.get_state()

        done = False
        total_reward = 0
        steps = 0
        while not done:
            state_tensor = torch.tensor([state], dtype=torch.float32)
            with torch.no_grad():
                logits = policy(state_tensor)
                action = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()

            # map back {0,1,2} → {-1,0,1}
            action = action - 1
            data.append(list(state))

            state, reward, done = env.step(action)

            total_reward += reward
            steps += 1


        print(f"Episode {ep+1}: survived {steps} steps, reward = {total_reward}")
        np.savetxt(f'data/episode_{ep+1}.csv', data, delimiter=',')

if __name__ == "__main__":
    evaluate()
