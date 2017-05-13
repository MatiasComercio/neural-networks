filename = 'terrain_perceptron_net.mat';

global figure_error;
figure_error = figure;

% Include patterns && expected_outputs
terrain_data;

config = get_config('terrain_perceptron');

epsilon = config.epsilon;
gap_size = config.gap.size;
gap_eval = config.gap.eval;
eta = config.eta;
alpha = config.alpha;

layers_neurons = config.layers.neurons;
layers_hidden_g = config.layers.hidden.g;
layers_hidden_g_derivative = config.layers.hidden.g_derivative;
layers_last_g = config.layers.last.g;
layers_last_g_derivative = config.layers.last.g_derivative;

% Choose 10 random samples from the patterns
[patterns_sample, patterns_indexes] = datasample(patterns, ...
    columns(patterns), rows(patterns), 'Replace', false);
expected_outputs_sample = expected_outputs(:, patterns_indexes);

% Split patterns in train patterns and test patterns
[train_patterns, train_expected_outputs, ...
    test_patterns, test_expected_outputs] = ...
    split_train_and_test_patterns(patterns_sample, ...
    expected_outputs_sample);

% Create terrain network
layers = create_layers(layers_neurons, layers_hidden_g, ...
    layers_hidden_g_derivative, layers_last_g, layers_last_g_derivative);
with_epsilon_are_close_enough = @(expected_output, neural_output) ...
    (epsilon_are_close_enough(expected_output, neural_output, epsilon));
cost_function = @mean_square_error;
net = neural_network(layers, with_epsilon_are_close_enough, cost_function);

% Keep training the network until the error for the test_patterns
% approximations is negligible
count = 1;
finished = false;
tic;
%while ~finished
%    % Train and test the network
%    [net, train_memory] = net.train(net, train_patterns, ...
%        train_expected_outputs, test_patterns, test_expected_outputs, eta, alpha, gap, eval_gap);
%    [finished, test_memory] = test_network(net, test_patterns, ...
%        test_expected_outputs, max_error);
%    % Save current train and test memory
%    global_memory(count).train_memory = train_memory;
%    global_memory(count).test_memory = test_memory;
%    % Lower the used epsilon
%    epsilon = epsilon / 2;
%    net.are_close_enough = @(expected_output, neural_output) ...
%        (are_close_enough(expected_output, neural_output, epsilon));
%    count = count + 1;
%end
[net, last_epoch, last_eta] = net.train(net, train_patterns, train_expected_outputs, eta, alpha, gap_size, gap_eval);
toc;

save(filename, 'net', 'train_patterns', 'train_expected_outputs', ...
    'test_patterns', 'test_expected_outputs');
