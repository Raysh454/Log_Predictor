# An 8 node hidden layer neural network implemented from scratch
# it outputs the common log of any number.
# Sochastic gradient descent

import sys
import math
import json
import copy
import os, random
from table import lookup, lookup_table


# Since this is a 8 node hidden layer neural network,
# and each node has 2 parameters (weight and bias),
# we have 8 (hidden layer weight) + 8 (hidden layer bias) + 8 (output layer weight) + 1 (output layer bias) == 25 parameters.
# And one input x.

EPOCHS = 10000

PARAMETER_FILE = "parameters_8.json"
HIDDEN_NODE_COUNT = 8
parameters = json.load(open(PARAMETER_FILE))
hidden_parameters = parameters["hidden_layer"]  # List of (weight, bias) dicts
output_parameters = parameters["output_layer"]  # List of weights + bias

LEARNING_RATE = 0.01

# Stopping criteria
MIN_DELTA = 1e-10
STOP_PATIENCE = 10
DECAY_PATIENCE = 5
DECAY_FACTOR = 0.9

def safe_write_json(filename, data):
    tmp_filename = filename + ".tmp"
    with open(tmp_filename, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_filename, filename)  # atomic rename

def tanhActivation(x):
    return math.tanh(x)

def backprop(x, predicted, actual, node_outputs):
    global parameters, output_parameters, hidden_parameters
    change_in_output_w = []
    for i in range(HIDDEN_NODE_COUNT):
        change_in_output_w.append(2 * (predicted - actual) * node_outputs[i])
    change_in_output_b = 2 * (predicted - actual)

    change_in_hidden_w = []
    change_in_hidden_b = []

    for i in range(HIDDEN_NODE_COUNT):
        change_in_hidden_w.append(2 * (predicted - actual) * output_parameters['weights'][i] * (1 - node_outputs[i] ** 2) * x)
        change_in_hidden_b.append(2 * (predicted - actual) * output_parameters['weights'][i] * (1 - node_outputs[i] ** 2))

    # Update output layer parameters
    new_parameters = copy.deepcopy(parameters)
    for i in range(len(output_parameters['weights'])):
        new_parameters["output_layer"]["weights"][i] = output_parameters['weights'][i] - LEARNING_RATE * change_in_output_w[i]
    new_parameters["output_layer"]["bias"] = output_parameters["bias"] - LEARNING_RATE * change_in_output_b

    # Update hidden layer parameters
    for i in range(len(hidden_parameters)):
        new_parameters['hidden_layer'][i]['weight'] = hidden_parameters[i]['weight'] - LEARNING_RATE * change_in_hidden_w[i]
        new_parameters['hidden_layer'][i]['bias'] = hidden_parameters[i]['bias'] - LEARNING_RATE * change_in_hidden_b[i]

    safe_write_json(PARAMETER_FILE, new_parameters)
    
    parameters = new_parameters
    hidden_parameters = parameters["hidden_layer"]
    output_parameters = parameters["output_layer"]
 
    #with open('parameter_hist.txt', 'a') as hist:
    #    hist.write(json.dumps(new_parameters) + '\n')
    

def reduce_input(x):
    counter = 0
    while x >= 10:
        x /= 10
        counter += 1
    return x, counter

def run(inp):
    x, inc = reduce_input(inp)

    if x == 1:
        return 0 + inc, 0
    else:
        node_outputs = [tanhActivation(x * n["weight"] + n["bias"]) for n in hidden_parameters]
        predicted = sum(wi * ni for wi, ni in zip(output_parameters['weights'], node_outputs)) + output_parameters['bias']
        return predicted + inc, abs(predicted + inc - math.log10(inp))


def __main__():
    if len(sys.argv) > 2:
        print("Usage: python manual_nn.py [number]")
        print("Starts training if no number provided")
        sys.exit(1)

    if len(sys.argv) == 2:
        inp = float(sys.argv[1])
        val, err = run(inp)
        if val is None or err is None:
            print("Unexpected error")
            return
        print(f"Predicted: {val}, Error: {err}")
    else:
        training()

def split_data(data, training_data_percent=0.8):
    items = list(data.items())
    split_index = int(len(items) * training_data_percent)
    training_data = dict(items[:split_index])
    validation_data = dict(items[split_index:])
    return training_data, validation_data

def shuffle_data(data):
    items = list(data.items())
    random.shuffle(items)
    return dict(items)

def test(data):
    err_sum = 0
    for x in data:
        node_outputs = [tanhActivation(x * n["weight"] + n["bias"]) for n in hidden_parameters]
        predicted = sum(wi * ni for wi, ni in zip(output_parameters['weights'], node_outputs)) + output_parameters['bias']
        err_sum += (lookup(x) - predicted) ** 2
    mse = 1/len(data) * err_sum
    return mse

def training():
    global LEARNING_RATE
    lookup_table_shuffled = shuffle_data(lookup_table)
    training_data, testing_data = split_data(lookup_table_shuffled, 0.8)
    prev_mse = float('inf')
    decay_patience = 0
    stop_patience = 0
    for epoch in range(EPOCHS):
        training_data = shuffle_data(training_data)
        node_outputs = []
        err_sum = 0
        for x in training_data:
            node_outputs = [tanhActivation(x * n["weight"] + n["bias"]) for n in hidden_parameters]

            predicted = sum(wi * ni for wi, ni in zip(output_parameters['weights'], node_outputs)) + output_parameters['bias']

            backprop(x, predicted, lookup(x), node_outputs)

            err_sum += (lookup(x) - predicted) ** 2
        mse = 1/len(lookup_table) * err_sum

        # Decay learning rate
        if mse > prev_mse:
            decay_patience += 1

        if decay_patience >= DECAY_PATIENCE:
            LEARNING_RATE *= DECAY_FACTOR
            print(f"Decayed learning rate to {LEARNING_RATE:.20f}")
            decay_patience = 0

        # Early stopping
        improvement = abs(prev_mse - mse)
        if improvement < MIN_DELTA:
            stop_patience += 1
        else:
            stop_patience = 0

        if stop_patience >= STOP_PATIENCE:
            print("Early stopping triggered")
            break

        prev_mse = mse

        print(f"MSE: {mse:.20f}, Epoch: {epoch}")

    # Test
    mse = test(testing_data)
    print(f"Final MSE on testing data: {mse:.20f}")


if __name__ == "__main__":
    __main__()
