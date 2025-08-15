#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--hidden_layer", default=16, type=int, help="Size of the hidden layer.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate_initial", default=0.025, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.001, type=float, help="Final learning rate.")
parser.add_argument("--model", default="gym_cartpole_model.keras", type=str, help="Output model path.")


def main(args: argparse.Namespace) -> keras.Model | None:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Load the data
    data = np.loadtxt("gym_cartpole_data_augmented.txt")
    observations, labels = data[:, :-1], data[:, -1].astype(np.int32)

    # TODO: Create the model in the `model` variable. Note that
    # the model can perform any of:
    # - binary classification with 1 output and sigmoid activation;
    # - two-class classification with 2 outputs and softmax activation.
    model = keras.Sequential([
        keras.layers.Dense(args.hidden_layer, activation="relu"),
        keras.layers.Dense(2, activation="softmax"),
    ])

    decay_steps = int(len(data) / args.batch_size) * args.epochs
    learning_rate = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.learning_rate_initial,
        decay_steps=decay_steps,
        alpha=args.learning_rate_final/args.learning_rate_initial
    )
    opt = keras.optimizers.AdamW(learning_rate=learning_rate)
    # TODO: Prepare the model for training using the `model.compile` method.
    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    # Train the model.
    model.fit(observations, labels, batch_size=args.batch_size, epochs=args.epochs)

    # Save the model, without the optimizer state.
    model.save(args.model)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
