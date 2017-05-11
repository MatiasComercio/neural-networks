function plot_error_bars( expected_outputs, outputs )
    global figure_error;
    figure(figure_error)
    subplot(1,2,1);
    
    bar(1:columns(outputs), abs(outputs - expected_outputs));
    drawnow;
end
