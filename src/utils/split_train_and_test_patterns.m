function [train_patterns, train_expected_outputs, test_patterns, test_expected_outputs] = split_train_and_test_patterns(patterns, expected_outputs)
    % Receives a set of Patterns and it's expected outputs, and splits them
    % in two sets: The first set (aproximately 70%) will be used to train
    % the network, the rest will be used to test the trained network
    
    patterns_amount = columns(patterns);
    train_patterns_amount = ceil(patterns_amount * 0.7); % Take aproximately 70% of the patterns to train the net
    test_patterns_amount = patterns_amount - train_patterns_amount;

    patterns_indexes = randperm(patterns_amount); % Select a specific random order

    train_patterns = zeros(rows(patterns), train_patterns_amount);
    train_expected_outputs = zeros(rows(expected_outputs), train_patterns_amount);
    test_patterns = zeros(rows(patterns), test_patterns_amount);
    test_expected_outputs = zeros(rows(expected_outputs), test_patterns_amount);

    % Select the patterns that will be used for training the net
    for i = 1:train_patterns_amount
        index = patterns_indexes(i);
        train_patterns(:,i) = patterns(:,index);
        train_expected_outputs(:,i) = expected_outputs(:, index);
    end

    % Select the patterns that will be used for testing the trained net
    for i = 1:test_patterns_amount
        test_pattern_i_offset = train_patterns_amount + i;
        pattern_index = patterns_indexes(test_pattern_i_offset);
        test_patterns(:,i) = patterns(:, pattern_index);
        test_expected_outputs(:,i) = expected_outputs(:,pattern_index);
    end

end

