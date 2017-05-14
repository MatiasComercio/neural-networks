function plot_original_patterns(net, original_patterns, original_expected_outputs)
    
    global figure_error_2;
    figure(figure_error_2)
    [outputs, ~] = solve_patterns(net, original_patterns);
    
    % Compare expected and estimated values for original patterns
    subplot(1,2,1);
    plot3(original_patterns(1,:), original_patterns(2,:), ...
        original_expected_outputs, 'ro');
    hold on;
    plot3(original_patterns(1,:), original_patterns(2,:), outputs, ...
        'b*');
    hold off;
    
    % Plot Estimated Surface
    subplot(1,2,2);
    [X, Y] = meshgrid(min(original_patterns(1,:)):0.1:max(original_patterns(1,:)), ...
        min(original_patterns(2,:)):0.1:max(original_patterns(2,:)));
    Z = griddata(original_patterns(1,:), original_patterns(2,:), outputs, X, Y);
    surf(X, Y, Z);
end
