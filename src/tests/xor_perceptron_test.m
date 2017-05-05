pattern = [
  1;
  1;
];

expected_output = -1;

eta = 0.5;

% Define which unit functions are going to be used
unit_function = 'non_linear_tanh';
g = str2func(strcat(unit_function, '_g'));
g_derivative = str2func(strcat(unit_function, '_g_derivative'));
are_close_enough = str2func(strcat(unit_function, '_are_close_enough'));
cost_function = @mean_square_error;
unit_functions.g = g;
unit_functions.g_derivative = g_derivative;
unit_functions.are_close_enough = are_close_enough;
unit_functions.cost_function = cost_function;

net = neural_network([rows(pattern), 2, rows(expected_output)], unit_functions);
net.layers(1).weights = [-.35, .15, .20; -.35, .25, .30];
net.layers(2).weights = [-.60, .40, .45];

% Get the current pattern's output with the current weights
[output, solve_memory] = net.solve(net, pattern);

% Fix weights for the current pattern
[layers, fix_memory] = net.fix(net.layers, expected_output, output, ...
  net.unit_functions.g_derivative, solve_memory, eta);

w_1_new = [-0.2751, 0.0751, 0.1251; -0.2854, 0.1854, 0.2354];
w_2_new = [-0.3050, 0.2217, 0.2387];

delta_w_1 = ones(2, 3) .* 1e-4;
delta_w_2 = ones(1, 3) .* 1e-4;

assert(all(all(abs(layers(1).weights - w_1_new) < delta_w_1)), ...
    '[FAIL] - w_1_new')
assert(all(all(abs(layers(2).weights - w_2_new) < delta_w_2)), ...
    '[FAIL] - w_2_new')
