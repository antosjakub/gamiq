#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--render", default=False, action="store_true", help="Render during evaluation")
parser.add_argument("--model", default="gym_cartpole_model.keras", type=str, help="Output model path.")


def evaluate_model(
    model: keras.Model, seed: int = 42, episodes: int = 10, render: bool = False, report_per_episode: bool = False
) -> float:
    """
    Evaluate the given model on CartPole-v1 environment.
    Returns the average score achieved on the given number of episodes.
    """
    import gymnasium as gym

    # Create the environment
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    env.reset(seed=seed)
    #env.reset(seed=seed, options={"low": -1.0, "high": 1.0})
    #env_unwrapped = env.unwrapped
    #env_unwrapped.theta_threshold_radians = np.deg2rad(40) 

    # Evaluate the episodes
    total_score = 0
    for episode in range(episodes):
        observation, score, done = env.reset()[0], 0, False
        observation = 1000*observation
        print(observation)
        
        while not done:
            prediction = model.predict_on_batch(observation[np.newaxis])[0]
            if len(prediction) == 1:
                action = 1 if prediction[0] > 0.5 else 0
            elif len(prediction) == 2:
                action = np.argmax(prediction)
            else:
                raise ValueError("Unknown model output shape, only 1 or 2 outputs are supported")

            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated

        total_score += score
        if report_per_episode:
            print("The episode {} finished with score {}.".format(episode + 1, score))
    return total_score / episodes


def main(args: argparse.Namespace) -> keras.Model | None:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)

    # Evaluate the model and optionally render the animation.
    model = keras.models.load_model(args.model, compile=False)

    score = evaluate_model(model, seed=args.seed, render=args.render, report_per_episode=True)
    print("The average score was {}.".format(score))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
