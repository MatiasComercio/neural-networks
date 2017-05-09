% Include patterns && expected_outputs
terrain_data;

eta = 0.05;
epsilon = .1;
alpha = 0.9;
gap = 100;
max_error = 5;

% Choose 10 random samples from the patterns
[patterns_sample, patterns_indexes] = ...
    datasample(patterns, 10, rows(patterns), 'Replace', false);
expected_outputs_sample = expected_outputs(:, patterns_indexes);

% Split patterns in train patterns and test patterns
[train_patterns, train_expected_outputs, ...
    test_patterns, test_expected_outputs] = ...
    split_train_and_test_patterns(patterns_sample, ...
    expected_outputs_sample);

% Create terrain network
with_epsilon_are_close_enough = @(expected_output, neural_output) ...
    (epsilon_are_close_enough(expected_output, neural_output, epsilon));
cost_function = @mean_square_error;
layers = create_all_non_linear_layers...
    ([rows(patterns), 7, rows(expected_outputs)]);
net = neural_network(layers, with_epsilon_are_close_enough, cost_function);

% Keep training the network until the error for the test_patterns
% approximations is negligible
count = 1;
finished = false;
while ~finished
    % Train and test the network
    [net, train_memory] = net.train(net, train_patterns, ...
        train_expected_outputs, eta, alpha, gap);
    [finished, test_memory] = test_network(net, test_patterns, ...
        test_expected_outputs, max_error);
    % Save current train and test memory
    global_memory(count).train_memory = train_memory;
    global_memory(count).test_memory = test_memory;
    % Lower the used epsilon
    epsilon = epsilon / 2;
    net.are_close_enough = @(expected_output, neural_output) ...
        (are_close_enough(expected_output, neural_output, epsilon));
    count = count + 1;
end

% TODO: Test only. Plots the error progress along
%   the whole network training
% What we are doing below is to pass each global_memory array element to
%   the first `arrayfun` function parameter, that computs the length of
%   the train_memory element of that specific struct, and then sum the
%   lengths of all those train_memories
train_memory_total_length = ...
    sum(arrayfun(@(memory_struct) numel(memory_struct.train_memory), ...
    global_memory));
scatter(1:train_memory_total_length, ...
    [global_memory(1:end).train_memory(1:end).global_error])


