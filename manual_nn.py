# An 8 node hidden layer neural network implemented from scratch
# it outputs the common log of any number.
# Sochastic gradient descent

import sys
import math
import json
import copy
import os
from table import lookup, lookup_table


# Since this is a 8 node hidden layer neural network,
# and each node has 2 parameters (weight and bias),
# we have 8 (hidden layer weight) + 8 (hidden layer bias) + 8 (output layer weight) + 1 (output layer bias) == 25 parameters.
# And one input x.

EPOCHS = 370

HIDDEN_NODE_COUNT = 8
parameters = json.load(open("parameters.json"))
hidden_parameters = parameters["hidden_layer"]  # List of (weight, bias) dicts
output_parameters = parameters["output_layer"]  # List of weights + bias

LEARNING_RATE = 0.001

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

    safe_write_json("parameters.json", new_parameters)
    
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
        return 0, 0
    else:
        node_outputs = [tanhActivation(x * n["weight"] + n["bias"]) for n in hidden_parameters]
        predicted = sum(wi * ni for wi, ni in zip(output_parameters['weights'], node_outputs)) + output_parameters['bias']
        return predicted + inc, abs(predicted - math.log10(x))


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


def training():
    global LEARNING_RATE
    for epoch in range(EPOCHS):
        node_outputs = []
        err_sum = 0
        for x in lookup_table:
            node_outputs = [tanhActivation(x * n["weight"] + n["bias"]) for n in hidden_parameters]

            predicted = sum(wi * ni for wi, ni in zip(output_parameters['weights'], node_outputs)) + output_parameters['bias']

            backprop(x, predicted, lookup(x), node_outputs)

            err_sum += (lookup(x) - predicted) ** 2
        if epoch % 50 == 0:
            LEARNING_RATE *= 0.9  # Decay learning rate
        mse = 1/len(lookup_table) * err_sum
        print(f"MSE: {mse}, Epoch: {epoch}")


__main__()
