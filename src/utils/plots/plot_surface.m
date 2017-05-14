function plot_surface( patterns, outputs, detail )
    global figure_error_2;
    figure(figure_error_2)
    subplot(1,2,2);

    [X, Y] = meshgrid(min(patterns(1,:)):detail:max(patterns(1,:)), ...
        min(patterns(2,:)):detail:max(patterns(2,:)));
    Z = griddata(patterns(1,:), patterns(2,:), outputs, X, Y);
    %plot3(patterns(1,:), patterns(2,:), outputs, 'ro');
    %hold on;
    surf(X, Y, Z);
end
