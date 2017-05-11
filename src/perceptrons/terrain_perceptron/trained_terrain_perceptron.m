filename = 'terrain_perceptron_net.mat';

load(filename);

patterns = create_rand_matrix(-3, 3, 2, 10000);

tic;
[outputs, memories] = solve_patterns(net, patterns);
toc;

original_patterns = horzcat(train_patterns, test_patterns);
original_expected_outputs = horzcat(train_expected_outputs, ...
    test_expected_outputs);

plot_original_and_solved(original_patterns, original_expected_outputs, ...
    patterns, outputs);
plot_surface(patterns, outputs, 0.1);
