function [train_patterns, train_expected_outputs, test_patterns, test_expected_outputs] = choose_train_patterns(patterns, expected_outputs, patterns_amount)
    % Receives a set of Patterns and it's expected outputs, and splits them
    % in two sets: The first set (aproximately 70%) will be used to train
    % the network, the rest will be used to test the trained network
    
    train_patterns_amount = ceil(patterns_amount * 0.7); % Take aproximately 70% of the patterns to train the net
    test_patterns_amount = patterns_amount - train_patterns_amount;

    patterns_indexes = randperm(patterns_amount); % Select a specific random order

    train_patterns = zeros(2, train_patterns_amount);
    train_expected_outputs = zeros(1, train_patterns_amount);
    test_patterns = zeros(2, test_patterns_amount);
    test_expected_outputs = zeros(1, test_patterns_amount);

    % Select the patterns that will be used for training the net
    for i = 1:train_patterns_amount
        index = patterns_indexes(i);
        train_patterns(:,i) = patterns(:,index);
        train_expected_outputs(i) = expected_outputs(index);

    end

    % Select the patterns that will be used for testing the trained net
    for i = train_patterns_amount+1:patterns_amount
        index = patterns_indexes(i);
        test_patterns(:,i-train_patterns_amount) = patterns(:,index);
        test_expected_outputs(i-train_patterns_amount) = expected_outputs(index);
    end

end

