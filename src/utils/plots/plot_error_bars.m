function plot_error_bars( outputs, expected_outputs )
    figure;
    bar(1:length(outputs), abs(outputs - expected_outputs));
end
