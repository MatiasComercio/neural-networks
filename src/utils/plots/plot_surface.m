function plot_surface( patterns, outputs, detail )
    figure;
    [X, Y] = meshgrid(min(patterns(1,:)):detail:max(patterns(1,:)), ...
        min(patterns(2,:)):detail:max(patterns(2,:)));
    Z = griddata(patterns(1,:), patterns(2,:), outputs, X, Y);
    surf(X, Y, Z);
end
