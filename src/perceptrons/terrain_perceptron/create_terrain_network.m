function net = create_terrain_network(patterns, expected_outputs, epsilon)

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
unit_functions.epsilon = epsilon;

net = neural_network([rows(patterns), 7, rows(expected_outputs)], unit_functions);

end

