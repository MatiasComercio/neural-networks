filename = 'terrain_perceptron_net.mat';

load(filename);
patterns = create_rand_matrix(-3, 3, 2, 10000);

original_patterns = horzcat(train_patterns, test_patterns);
original_expected_outputs = horzcat(train_expected_outputs, ...
    test_expected_outputs);

plot_surface_comparison(net, patterns, original_patterns, original_expected_outputs);

