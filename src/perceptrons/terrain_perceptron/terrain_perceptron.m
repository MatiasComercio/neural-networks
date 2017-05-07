% Include patterns && expected_outputs
terrain_data;

epsilon = .1;
eta = 0.05;
max_error = 5;

% Choose 20 random samples from the patterns
[patterns_sample, patterns_indexes] = datasample(patterns, 10, rows(patterns), 'Replace', false);
expected_outputs_sample = expected_outputs(:, patterns_indexes);

% Split patterns in train patterns and test patterns
[train_patterns, train_expected_outputs, test_patterns, test_expected_outputs] = split_train_and_test_patterns(patterns_sample, expected_outputs_sample);

finished = false;

net = create_terrain_network([rows(patterns), 7, rows(expected_outputs)], epsilon);

% Keep training the network until the error for the test_patterns
% approximations is negligible
count = 1;
while ~finished
    % Train and test the network
    [net, train_memory] = net.train(net, train_patterns, train_expected_outputs, eta);
    [finished, test_memory] = test_network(net, test_patterns, test_expected_outputs, max_error);
    % Save current train and test memory
    global_memory(count).train_memory = train_memory;
    global_memory(count).test_memory = test_memory;
    % Lower the used epsilon
    net.unit_functions.epsilon = net.unit_functions.epsilon / 2;
    count = count + 1;
end

% TODO: Test only. Plots the error progress along the whole network training
%train_memory_total_length = sum(arrayfun(@(memory_struct) numel(memory_struct.train_memory), global_memory));
%scatter(1:train_memory_total_length, [global_memory(1:end).train_memory(1:end).global_error])


