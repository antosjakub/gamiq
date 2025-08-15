# Gymnasium cartpole-v1 challange

## Problem statement

Solve the CartPole-v1 environment from the Gymnasium library, utilizing only provided supervised training data. The data is available in `gym_cartpole_data.txt` file, each line containing one observation (four space separated floats) and a corresponding action (the last space separated integer).

The solution to this task should be a model which passes evaluation on random inputs. This evaluation can be performed by running the `evaluate_model.py` script and optionally rendering if --render option is provided.

In order to pass, you must achieve an average reward of at least 475 on 100 episodes. Your model should have either one or two outputs (i.e., using either sigmoid or softmax output function).

When designing the model, you should consider that the size of the training data is very small and the data is quite noisy.


## Solution

Steps:

1) Create more training data by augmenting the existing dataset:

    ```python data_augmentation.py```

2) Train and save the model:

    ```python gym_cartpole.py```

3) Test the model on random data and see it completely smash the minimal required score of 475 / 500:

    ```python evaluate_model.py```

- (Optional) To see how the model is doing on the evaluation, render the simulation with:

    ```python evaluate_model.py --render```

off the screen:
python gym_cartpole.py --learning_rate_initial=0.05 --learning_rate_final=0.05 --hidden_layer=4