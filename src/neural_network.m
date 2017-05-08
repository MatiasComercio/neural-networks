% Conventions:
% - weights: row vector
% - pattern: column vector
% - patterns: matrix with each column being a pattern
% - extended_patterns: patterns with the first row being -1 (constant bias factor)
% - eta: learning rate
% - expected_outputs: row vector containing the expected output of patterns(i) at 
%     position i.
% - neural_output: row vector containint the resulting output of patterns(i) at
%     position i, that was obtained using the current `g` function and the current
%     weigths vector 
%
% Notes
% - Structures cannot be modified inside functions. That's why we need to
%   reassign them.

function net = neural_network(neurons_per_layer, are_close_enough, ...
    cost_function)
% neurons_per_layer: array that, for each position indicating a layer, 
%     specifies how many neurons should that layer have and the unit
%     functions to be used on that layer. The input and
%     output layers must be included too (so, if you want to create a
%     neural network of M layers, neurons_per_layer should have length M+1).
%   For example (only representing the number of neurons):
%   	[ 2, 1 ] represents a single layer perceptron with 
%       2 neurons on the input layer and one output neuron
%     [ 4, 2, 3 ] represents a multilayer perceptron with 2 neurons on the
%       input layer, one hidden layer with 2 neurons 
%       and 3 neurons on the output layer
%   Both parameters (number of neurons and unit functions for each layer)
%     should be specified using a structure with the following fields:
%   - neurons: amount of neurons for the current layer
%   - g: activation function
%   - g_derivative: g derivative function
% are_close_enough: network output evaluation function
% cost_function: used to calculate the error. Example: @mean_square_error

  net = struct;
  net.layers = create_layers(neurons_per_layer);
  net.cost_function = cost_function;
  net.are_close_enough = are_close_enough;
  net.train = @train;
  net.solve = @(net, pattern) solve(net.layers, pattern);
  % This is made public for testing purposes only
  net.fix = @fix;
end

function layers = create_layers(neurons_per_layer)
% neurons_per_layer = array of length M + 1
% Example of usage:
%   >> layers = create_layers([2, 1]) 
%   => layers = struct {
%         1: {
%           weights = [ rand_in_range(min, max, 1, 3) ]
%         }
%       }

  % n_layers = M neural network
  n_layers = length(neurons_per_layer) - 1;
  % Preallocate space for all layers
  layers(n_layers).weights = 0;
  % Start from the first layer that will have weights, i.e., associations
  %   between two of the given layers. That is, layers are constructed
  %   using two sets of neurons (the current and the next) and its
  %   connections, and all this information is stored in the weights matrix
  % What we are actually doing is compressing two neuron layers (current
  %   and next) specified by the user into one layer with more information
  %   (two neuron layers and their weighted connections)
  % See the example above and/or check the files at `docs` folder
  for m = 1:n_layers
    % Input size for current layer = neurons of current layer + 1 for 
    %   constant for bias neuron
    input_length = neurons_per_layer(m).neurons + 1;
    % Amount of neurons on the current layer for processing the input
    % Note that the neurons of the current layer are the ones specified
    %   that the user specified for the next layer, as we noted before
    user_layer = neurons_per_layer(m+1);
    n_neurons = user_layer.neurons;
    layers(m).weights = create_weights_matrix(n_neurons, input_length);
    layers(m).g = user_layer.g;
    layers(m).g_derivative = user_layer.g_derivative;
  end
end

function [net, train_memory] = train(net, patterns, expected_outputs, eta)
  layers = net.layers;

  % Train until expected outputs match the neural outputs
  finished = false;
  epoch_i = 1;
  while ~finished
    % Train one complete epoch
    [layers, neural_outputs, epoch_memory] = ...
        epoch(layers, patterns, expected_outputs, eta);
    % Calculate global error
    global_error = net.cost_function(expected_outputs, neural_outputs);
    % Determine whether the outputs match
    finished = have_to_finish(expected_outputs, neural_outputs, ...
        net.are_close_enough);
    % Save current epoch parameters
    train_memory(epoch_i).layers = layers;
    train_memory(epoch_i).neural_outputs = neural_outputs;
    train_memory(epoch_i).epoch_memory = epoch_memory;
    train_memory(epoch_i).global_error = global_error;
    epoch_i = epoch_i + 1;
  end
  net.layers = layers;
end

function [output, memory] = solve(layers, pattern)
% layers: layers of a neural network
% pattern: column vector representing the input of the neural network
%   It should have the declared length (same as when creating the neural
%   network)
% Return
%   - output: the current output (column vector) of the neural network
%             for the given pattern
%   - memory: inputs (and outputs) of each layer of the network
%             for the given pattern
  M = length(layers);
  % M inputs + 1 output to store the output on that same array
  memory(M + 1).V = 0;
  memory(1).V = pattern;
  for m = 1:M
     layer = layers(m);
     % Add the current bias factor as input too
     input = [ -1 ; memory(m).V ];
     % Output of current layer is the input of the next one
     memory(m+1).V = layer.g(layer.weights * input);
  end
  output = memory(M + 1).V;
end


%% Below functions should act as `private` ones

function [layers, memory] = fix(layers, expected_output, ...
    output, solve_memory, eta)
  M = length(layers);
  g_output = output;
  % Calculate the delta for the last layer as a different case
  memory(M).deltas = layers(M).g_derivative(g_output) .* ...
      (expected_output - output);
  % Calculate deltas for all previous layers
  for m = M-1:-1:1
    % Output of the current layer is the input of the following
    g_output = solve_memory(m+1).V;
    % Remove first column (i.e.: connections with the bias factor neuron)
    weights = layers(m+1).weights(:,2:end);
    deltas = memory(m+1).deltas;
    memory(m).deltas = layers(m).g_derivative(g_output) .* ...
        (weights.' * deltas);
  end
  % Fix all layers's weights
  for m = 1:M
    weights = layers(m).weights;
    deltas = memory(m).deltas;
    % Recall to add the current bias factor as input too
    input = [ -1 ; solve_memory(m).V ];
    layers(m).weights = weights + eta .* deltas * input.';
  end
end

function weights = create_weights_matrix(n_neurons, input_length)
% Weights matrix will have `n_neurons` rows and `input_length` columns
%   i.e.: all columns of a row represent all weights for a neuron
% See files at `docs` folder for more details
  weights = create_rand_matrix(-0.5, 0.5, n_neurons, input_length); % TODO: weights range limits as input parameters
end

function [layers, neural_outputs, memory] = epoch(layers, patterns, ...
    expected_outputs, eta)
  patterns_amount = columns(patterns);
  % Create a permutation to get a different sample patterns
  % Recall each pattern is a column vector, so, the number of columns of
  %   the extended_patterns matrix is the number of total patterns
  sample_patterns_indexes = randperm(patterns_amount);
  % Initialize the memory row vector
  memory(1).neural_output = 0;
  memory = save_to_epoch_memory(memory, patterns_amount, 0, 0, 0);
  % Train with each pattern in the specified random order
  for i = sample_patterns_indexes
    % Get the current sample pattern and its expected output
    pattern = patterns(:,i);
    expected_output = expected_outputs(:,i);
    % Get the current pattern's output with the current weights
    [output, solve_memory] = solve(layers, pattern);
    % Fix weights for the current pattern
    [layers, fix_memory] = fix(layers, expected_output, output, ...
      solve_memory, eta);
    % Save current pattern training results
    memory = save_to_epoch_memory(memory, i, output, solve_memory, ...
        fix_memory);
  end
  neural_outputs = [memory(:).neural_output];
end

function mem = save_to_epoch_memory(mem, i, output, solve_mem, fix_mem)
  mem(i).neural_output = output;
  mem(i).solve_memory = solve_mem;
  mem(i).fix_memory = fix_mem;
end

function outputs_match = have_to_finish(expected_outputs, ...
    neural_outputs, are_close_enough)
  outputs_match = true;
  for i = 1:columns(expected_outputs)
    outputs_match = outputs_match && ...
        are_close_enough(expected_outputs(i), neural_outputs(i));
    if ~ outputs_match
      return;
    end
  end
end

