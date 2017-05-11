function plot_train_test_error( epoch_i,  train_global_error)
    global figure_error;
    figure(figure_error)
    subplot(1,2,2);
    
    % Plot global training and test error
    scatter(epoch_i, train_global_error, 'b*');
    hold on;
    drawnow;
end

