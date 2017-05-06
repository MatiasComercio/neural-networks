% Include patterns && expected_outputs
terrain_data;

epsilon = .1;

%patterns_amount = columns(patterns); % Total ammount is 441

% TEST ONLY
patterns_amount = 50; % (Use 50 out of 441 patterns, since program won't finish with 441 patterns)

% Split patterns in train patterns and test patterns
[train_patterns, train_expected_outputs, test_patterns, test_expected_outputs] = choose_train_patterns(patterns, expected_outputs, patterns_amount);

finished = false;

% Keep training the network until the error for the test_patterns
% approximations is negligible
while ~finished 
    [net, train_memory] = train_network(train_patterns, train_expected_outputs, epsilon);
    finished = test_network(net, test_patterns, test_expected_outputs);
    epsilon = epsilon / 2;    
end

% scatter(1:length(train_memory), [train_memory(1:end).global_error])
