# try different parameters
# - learning rate, hidden layer size, episodes
# - do not store model
# - do not print during training
# - run evaluation and print some results


# python reinforce.py 
# --hidden_dim=32 --episodes=1000 --gamma=0.99 --learning_rate=0.01
# --save_model=False --print_when_training=False --evaluate=False --evaluate_episodes=10 --print_when_evaluating=False --save_eval_episodes=False

# script_a.py
import subprocess

# Arguments to pass
script = ["python", "reinforce.py"]
model_behavior = ["--evaluate", "--evaluate_episodes", "10"]

hidden_dim_options = [8, 32, 128]
episodes_options = [500, 1500, 4500]
gamma_options = [0.9, 0.95, 0.99]
learning_rate_options = [0.001, 0.01, 0.1]
#hidden_dim_options = [8, 32]
#episodes_options = [500, 1000]
#gamma_options = [0.99]
#learning_rate_options = [0.01]

with open("results_gs.txt", "w") as f:
    for hidden_dim in hidden_dim_options:
        for episodes in episodes_options:
            for gamma in gamma_options:
                for learning_rate in learning_rate_options:

                    model_params = ["--hidden_dim", str(hidden_dim), "--episodes", str(episodes), "--gamma", str(gamma), "--learning_rate", str(learning_rate)]
                    args = script + model_params + model_behavior
                    print(model_params)
                    f.write(f"{model_params}\n")
                    f.flush()  # make sure it’s written immediately
                    result = subprocess.run(args, capture_output=True, text=True)
                    print(result.stdout)
                    f.write(f"{result.stdout}\n")
                    f.flush()  # make sure it’s written immediately
