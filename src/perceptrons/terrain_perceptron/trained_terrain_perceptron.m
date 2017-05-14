filename = 'terrain_perceptron_net.mat';
load(filename);
patterns = create_rand_matrix(-3, 3, 2, 10000);

[outputs, memories] = solve_patterns(net, patterns);

original_patterns = horzcat(train_patterns, test_patterns);
original_expected_outputs = horzcat(train_expected_outputs, ...
    test_expected_outputs);

plot_original_and_solved(original_patterns, original_expected_outputs, patterns, outputs);
plot_surface(patterns, outputs, 0.1);

% Calculate the net generalization capacity for the test patterns
config = get_config('terrain_perceptron');
epsilon = config.epsilon;

[outputs, ~] = solve_patterns(net, test_patterns);
close_enough = 0;
for i=1:length(outputs)
    if( abs(test_expected_outputs(i) - outputs(i)) < epsilon )
        close_enough = close_enough + 1;
    end
end
generalization_capacity = close_enough / length(test_patterns)
