function plot_surface_comparison(net, patterns, original_patterns, original_expected_outputs)
    [outputs, memories] = solve_patterns(net, patterns);
    plot_original_and_solved(original_patterns, original_expected_outputs, ...
        patterns, outputs);
    plot_surface(patterns, outputs, 0.1);
end
