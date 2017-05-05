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

function neural_network = get_neural_network(neurons_per_layer, ...
  unit_functions)
% neurons_per_layer: array containing the number of neurons for each
%   layer of the neural network.
%   For example:
%   	[ 2, 1 ] represents a single layer perceptron with 
%       2 neurons on the input layer and one output neuron
%     [ 4, 2, 3 ] represents a multilayer perceptron with 2 neurons on the
%       input layer, one hidden layer with 2 neurons 
%       and 3 neurons on the output layer
%
% unit_functions: structure with the following definition
%   unit_function.g: activation function
%   unit_function.delta: delta function
%   unit_function.cost_function: cost function
%   unit_function.are_close_enough: network output evaluation function

  neural_network.neurons_per_layer = neurons_per_layer;
  neural_network.unit_functions = unit_functions;
  neural_network.train = @(patterns, expected_outputs, eta) ...
      train(neural_network, patterns, expected_outputs, eta);
  neural_network.solve = @solve; % TODO
  neural_network.fix = @fix; % TODO
  neural_network.is_trained = false;
end

function [weights, global_errors] = train(neural_network, ... 
    patterns, expected_outputs, eta) % +++X_DOING +++X_FIX 

  % Build the extended patterns
  bias_constant_factors = ones(1, columns(patterns)) .* -1;
  extended_patterns = [ 
    bias_constant_factors;
    patterns;
  ];

  % Create the weights row vector with values between [-0.5; 0.5]
  weights = initialize_weights(rows(extended_patterns));
  % Train until expected outputs match the neural outputs
  finished = false;
  era_counter = 0;
  global_errors = [];
  while ~finished
    % Train one complete era
    [neural_outputs, weights] = era(weights, extended_patterns, ....
        expected_outputs, eta, neural_network);
    global_errors = [global_errors; ...
        cost_function(expected_outputs, neural_outputs)];
    % Determine whether the outputs match
    finished = do_outputs_match(expected_outputs, neural_outputs, ...
        are_close_enough);
    era_counter = era_counter + 1;
  end
end

function [neural_network, neural_outputs, patterns_layers_memory] = ...
        era(neural_network, extended_patterns, expected_outputs, eta) 
  % Create a permutation to get a different sample patterns
  % Recall each pattern is a column vector, so, the number of columns of
  %   the extended_patterns matrix is the number of total patterns
  sample_patterns_indexes = randperm(columns(extended_patterns));
  % Initialize the output row vector
  neural_outputs = zeros(1, columns(expected_outputs));
  patterns_layers_memory = zeros(1, columns(expected_outputs));
  % Train with each pattern in the specified random order
  for i = sample_patterns_indexes
    % Get the current sample pattern
    pattern = extended_patterns(:,i);
    % Solve the current pattern for the current weights
    [ neural_network, neural_outputs(i), patterns_layers_memory(i) ] = ...
        neural_network.solve(neural_network, pattern);
    % Fix network weights for the current pattern
    neural_network = neural_network.fix(neural_network, expected_outputs(i), ...
        neural_outputs(i), pattern, eta);
  end
end

function ret = initialize_weights(length)
  ret = random_row_in_range(-0.5, 0.5, length);
end

function is_expected = do_outputs_match(expected_outputs, ...
    neural_outputs, are_close_enough)
  is_expected = true;
  for i = 1:length(expected_outputs)
    is_expected = is_expected && ...
        are_close_enough(expected_outputs(i), neural_outputs(i));
    if ~ is_expected
      return;
    end
  end
end




% TODO: Move this functions to `utility` or `helper` folder

function ret = random_row_in_range(min, max, length)
  % min: inclusive
  % max: inclusive
  % length: row vector length
  ret = (max-min).*rand(1, length) + min;
end
