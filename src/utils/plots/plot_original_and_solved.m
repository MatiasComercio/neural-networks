function plot_original_and_solved( original_patterns, ...
    original_outpoouts, solved_patterns, solved_outputs )

    figure;
    plot3(original_patterns(1,:), original_patterns(2,:), ...
        original_outpoouts, 'ro');
    hold on;
    plot3(solved_patterns(1,:), solved_patterns(2,:), solved_outputs, ...
        'b*');
end
