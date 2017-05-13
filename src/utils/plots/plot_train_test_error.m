function plot_train_test_error( epoch_i,  train_global_error, test_global_error)
    global figure_error;
    figure(figure_error)
    subplot(1,2,2);
    scatter(epoch_i, train_global_error, 'b*');
    hold on;
    scatter(epoch_i, test_global_error, 'r+');
    hold on;
    drawnow;
end

