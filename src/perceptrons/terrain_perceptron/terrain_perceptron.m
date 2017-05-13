

global figure_error;
figure_error = figure;

config = get_config('terrain_perceptron');

filename = config.filename;

patterns = config.patterns;
expected_outputs = config.expected_outputs;
data_size = config.data_size;
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

% Choose random samples from the patterns
[patterns_sample, patterns_indexes] = datasample(patterns, ...
    data_size, rows(patterns), 'Replace', false);
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

tic;
[net, last_epoch, test_epoch, eta] = net.train(net, train_patterns, ...
    train_expected_outputs, test_patterns, test_expected_outputs, eta, ...
    alpha, gap_size, gap_eval);
toc;

save(filename, 'net', 'train_patterns', 'train_expected_outputs', ...
    'test_patterns', 'test_expected_outputs');
