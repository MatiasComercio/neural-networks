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
  net.neurons_per_layer = neurons_per_layer;
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
    sigma_m = sqrt(input_length) ^ -1;
    layers(m).weights = create_weights_matrix(n_neurons, input_length, ...
        sigma_m);
    layers(m).g = user_layer.g;
    layers(m).g_derivative = user_layer.g_derivative;
  end
end

function [ net, curr_epoch, test_epoch, eta ] = train( net, train_patterns, ...
    train_expected_outputs, test_patterns, test_expected_outputs, ...
    eta, original_alpha, gap, evaluate_gap )

    %global figure_error;
    %global figure_error_2;
    
    % Load configurable variables
    config = get_config('neural_network');
    net_save_period = config.net_save_period;
    error_bars_plot_period = config.error_bars_plot_period;
    
    % Choose a random set of patterns to better visualize how the network
    % is working
    random_patterns = create_rand_matrix(-3, 3, 2, 10000);
    original_patterns = horzcat(train_patterns, test_patterns);
    original_expected_outputs = horzcat(train_expected_outputs, ...
        test_expected_outputs);

    train_outputs_size = rows(train_expected_outputs);
    test_outputs_size = rows(test_expected_outputs);
    
    alpha = original_alpha;

    % Initial epoch values
    original_epoch.i = 0;
    original_epoch.layers = net.layers;
    original_epoch.memory = nan;
  
    % Analyze initial epoch results
    train_outputs = solve_all(original_epoch.layers, train_patterns, ...
        train_outputs_size);
    original_epoch.train_global_error = net.cost_function(train_expected_outputs, ...
        train_outputs);
    
    % Test initial epoch
    test_outputs = solve_all(original_epoch.layers, test_patterns, test_outputs_size);
	original_epoch.test_global_error = net.cost_function(test_expected_outputs, test_outputs);
    
    % Normalize Global Errors
    original_normalized_train_error = original_epoch.train_global_error/columns(train_patterns);
    original_normalized_test_error = original_epoch.test_global_error/columns(test_patterns);   
    
    % Plot initial epoch errors
    plot_train_test_error(original_epoch.i, original_normalized_train_error, ...
        original_normalized_test_error);    
    
    % Set last good train & test epochs to original epoch
    train_epoch = original_epoch;
    test_epoch = original_epoch;
    
    [prev_epoch, curr_epoch] = next_epoch(original_epoch);
    
    % Train until expected outputs match the training outputs
	finished = false;
    while ~finished
        % Train one complete epoch
        [curr_epoch.layers, curr_epoch.memory] = epoch(prev_epoch.layers, ...
            train_patterns, train_expected_outputs, prev_epoch.memory, eta, alpha);

        % Analyze epoch's training results
        train_outputs = solve_all(curr_epoch.layers, train_patterns, ...
            train_outputs_size);
        curr_epoch.train_global_error = net.cost_function(train_expected_outputs, ...
            train_outputs);
        
        % Test epoch
        test_outputs = solve_all(curr_epoch.layers, test_patterns, ...
            test_outputs_size);
        curr_epoch.test_global_error = net.cost_function(test_expected_outputs, ...
            test_outputs);
        
        gap_passed = (mod(curr_epoch.i, gap) == 0);
        if (gap_passed)
            
            % Difference with previous epoch training error
            curr_error_variation = curr_epoch.train_global_error - prev_epoch.train_global_error;
            error_variation(curr_epoch.i) = curr_error_variation;
            
            % Adapt eta and alpha according to global error change through this gap
            [eta, alpha, good_epoch] = evaluate_gap(curr_error_variation, eta, alpha, original_alpha, error_variation);
            
            % If this is not a good gap epoch go back to the last one
            if (~good_epoch)
                [prev_epoch, curr_epoch] = next_epoch(train_epoch);
                continue;
            end
            
            % Set train_epoch as this is the last good gap epoch
            train_epoch = curr_epoch;
        end
        
        % Normalize Global Errors
        prev_normalized_train_error = prev_epoch.train_global_error/columns(train_patterns);
        prev_normalized_test_error = prev_epoch.test_global_error/columns(test_patterns);
        curr_normalized_train_error = curr_epoch.train_global_error/columns(train_patterns);
        curr_normalized_test_error = curr_epoch.test_global_error/columns(test_patterns);
        
        % Plot current epoch errors
        plot_train_test_error(curr_epoch.i, curr_normalized_train_error, curr_normalized_test_error);
        if(mod(curr_epoch.i, error_bars_plot_period) == 0)
            plot_error_bars(train_expected_outputs, train_outputs);
        end
        
        % Save epoch if test error intersects train error
        if (sign(prev_normalized_train_error  - prev_normalized_test_error) ~= ...
                sign(curr_normalized_train_error - curr_normalized_test_error ) ...
                    || curr_normalized_train_error == curr_normalized_test_error ) 
            test_epoch = curr_epoch;
            
            % Save current figure in file
            %print(figure_error, 'terrain_perceptron_net_test', '-dpng')
            
            % Save current net in file
            %aux_layers = net.layers;
            %net.layers = test_epoch.layers;
            %save('terrain_perceptron_net_test.mat', 'net', 'train_patterns', ...
            %    'train_expected_outputs', 'test_patterns', 'test_expected_outputs');
            %net.layers = aux_layers;        
        end
        
        % Every 200 epoch, save current epoch
        if(mod(curr_epoch.i, net_save_period) == 0)
            aux_layers = net.layers;
            net.layers = test_epoch.layers;
            % Save current net in file
            save('terrain_perceptron_net_test.mat', 'net', 'train_patterns', ...
                'train_expected_outputs', 'test_patterns', 'test_expected_outputs');
            plot_surface_comparison(net, random_patterns, original_patterns, original_expected_outputs);
            net.layers = aux_layers;
        end
        
        alpha = original_alpha;
        
        % Determine whether the outputs match
        finished = have_to_finish(train_expected_outputs, train_outputs, ...
            net.are_close_enough);
        
        if ~finished
            [prev_epoch, curr_epoch] = next_epoch(curr_epoch);
        end
    end
    
    % Update net layers
    net.layers = curr_epoch.layers;
    plot_error_bars(train_expected_outputs, train_outputs);
    plot_surface_comparison(net, random_patterns, original_patterns, original_expected_outputs);
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

function [prev_epoch, curr_epoch] = next_epoch(epoch)
    prev_epoch = epoch;
    curr_epoch.i = prev_epoch.i + 1;
end

function [layers, memory] = fix(layers, expected_output, ...
    output, solve_memory, prev_fix_memory, eta, alpha)
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
    % Get the current deltas for all weights and save them to memory
    curr_deltas_weights = eta .* deltas * input.';
    memory(m).deltas_weights = curr_deltas_weights;
    % Calculate the new weights
    layers(m).weights = weights + curr_deltas_weights + ...
        prev_deltas_weights_percentage(prev_fix_memory, m, alpha);
  end
end

function weights = create_weights_matrix(n_neurons, input_length, sigma_m)
% Weights matrix will have `n_neurons` rows and `input_length` columns
%   i.e.: all columns of a row represent all weights for a neuron
% See files at `docs` folder for more details
  weights = create_rand_matrix(-sigma_m, sigma_m, n_neurons, input_length);
end

function [layers, memory] = epoch(layers, patterns, ...
    expected_outputs, prev_epoch_memory, eta, alpha)
  patterns_amount = columns(patterns);
  % Create a permutation to get a different sample patterns
  % Recall each pattern is a column vector, so, the number of columns of
  %   the extended_patterns matrix is the number of total patterns
  sample_patterns_indexes = randperm(patterns_amount);
  % Initialize the memory row vector
  memory(1).neural_output = 0;
  memory = save_to_epoch_memory(memory, patterns_amount, 0, 0, 0);
  % Initialize the prev_fix_memory
  prev_fix_memory = get_prev_fix_memory(prev_epoch_memory);
  % Train with each pattern in the specified random order
  for i = sample_patterns_indexes
    % Get the current sample pattern and its expected output
    pattern = patterns(:,i);
    expected_output = expected_outputs(:,i);
    % Get the current pattern's output with the current weights
    [output, solve_memory] = solve(layers, pattern);
    % Fix weights for the current pattern
    [layers, fix_memory] = fix(layers, expected_output, output, ...
      solve_memory, prev_fix_memory, eta, alpha);
    % Save current pattern training results
    memory = save_to_epoch_memory(memory, i, output, solve_memory, ...
        fix_memory);
    % Update the given prev_fix_memory
    prev_fix_memory = fix_memory;
  end
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

function prev_fix_memory = get_prev_fix_memory(prev_epoch_memory)
  prev_fix_memory = nan;
  if isstruct(prev_epoch_memory)
    prev_fix_memory = prev_epoch_memory.fix_memory;
  end
end

function ret = prev_deltas_weights_percentage(prev_fix_memory, ...
    layer_index, alpha)
  ret = 0;
  if isstruct(prev_fix_memory)
    ret = alpha .* prev_fix_memory(layer_index).deltas_weights;
  end
end

function neural_outputs = solve_all(layers, patterns, rows_output)
  patterns_amount = columns(patterns);
  neural_outputs = zeros(rows_output, patterns_amount);
  for i = 1:patterns_amount
    neural_outputs(:,i) = solve(layers, patterns(:, i));
  end
end
