% Define problem
V_1 = [
  -1;
  1;
];
expected_outputs = 1;
W_1 = [-.35, .15, .20; -.35, .25, .30];
W_2 = [-.60, .40, .45];
eta = 0.5;
alpha = 0.9;
gap = 1;
g = @tanh;
g_derivative = @(g) 1-g.^2 + .1; % \beta = 1
cost_function = @(a, b) sum(sum((a - b).^2));

% First epoch
% Forward
h_1 = W_1 * [ -1 ; V_1 ];
g_1 = g(h_1); V_2 = g_1;

h_2 = W_2 * [ -1 ; V_2 ];
g_2 = g(h_2); V_3 = g_2;

prev_global_error = cost_function(expected_outputs, V_3);

% Backward
delta_2 = g_derivative(g_2) .* (expected_outputs - V_3);
delta_1 = g_derivative(g_1) .* (W_2(:, 2:end).' * delta_2);

delta_weights_1_1 = eta .* (delta_1 * [ -1 ; V_1 ].');
delta_weights_1_2 = eta .* (delta_2 * [ -1 ; V_2 ].');
W_1_new = W_1 + delta_weights_1_1;
W_2_new = W_2 + delta_weights_1_2;

% Solve as it is
h_1 = W_1_new * [ -1 ; V_1 ];
g_1 = g(h_1); V_2 = g_1;

h_2 = W_2_new * [ -1 ; V_2 ];
g_2 = g(h_2); V_3 = g_2;

global_error = cost_function(expected_outputs, V_3);

% Second epcoch
% Save for display
W_1_1 = W_1_new;
W_1_2 = W_2_new;
% Update for next step
W_1 = W_1_new;
W_2 = W_2_new;
% Error should have decremented during this gap (global_error <
% prev_global_error should be true; you can check it out) => increment eta
eta = eta + 0.001;

% Forward
h_1 = W_1 * [ -1 ; V_1 ];
g_1 = g(h_1); V_2 = g_1;

h_2 = W_2 * [ -1 ; V_2 ];
g_2 = g(h_2); V_3 = g_2;

prev_global_error = cost_function(expected_outputs, V_3);

% Backward
delta_2 = g_derivative(g_2) .* (expected_outputs - V_3);
delta_1 = g_derivative(g_1) .* (W_2(:, 2:end).' * delta_2);

W_1_new = W_1 + eta .* (delta_1 * [ -1 ; V_1 ].') + ...
    alpha * delta_weights_1_1;
W_2_new = W_2 + eta .* (delta_2 * [ -1 ; V_2 ].') + ...
    alpha * delta_weights_1_2;

% Take these two matrixes (W_1_new & W_2_new): these should be the outputs
% of the test_memory weights for each layer for the second epoch;
% eta should have incremented by .001 for that epoch too

% Eta decrementation is not easy to check (we must force an error while
% training is occurring) and that's why this case can be manually checked
% when running a perceptron and checking how eta has changed during all the
% training epochs, recalling that its changes are related to the 
% specified gap
