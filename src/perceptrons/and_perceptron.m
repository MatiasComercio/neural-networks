% % Read input patterns
% patterns = [
%   1, 1,-1,-1;
%   1,-1, 1,-1;
% ];
% 
% % Read expected outputs
% expected_outputs = [
%   1, 1, 1, -1;
% ];

% Note that this patterns are row oriented => we have to transpose them
%   in order to use them
patterns = [
-1,-1,-1,-1,-1;
-1,-1,-1,-1,1;
-1,-1,-1,1,-1;
-1,-1,-1,1,1;
-1,-1,1,-1,-1;
-1,-1,1,-1,1;
-1,-1,1,1,-1;
-1,-1,1,1,1;
-1,1,-1,-1,-1;
-1,1,-1,-1,1;
-1,1,-1,1,-1;
-1,1,-1,1,1;
-1,1,1,-1,-1;
-1,1,1,-1,1;
-1,1,1,1,-1;
-1,1,1,1,1;
1,-1,-1,-1,-1;
1,-1,-1,-1,1;
1,-1,-1,1,-1;
1,-1,-1,1,1;
1,-1,1,-1,-1;
1,-1,1,-1,1;
1,-1,1,1,-1;
1,-1,1,1,1;
1,1,-1,-1,-1;
1,1,-1,-1,1;
1,1,-1,1,-1;
1,1,-1,1,1;
1,1,1,-1,-1;
1,1,1,-1,1;
1,1,1,1,-1;
1,1,1,1,1;
].';

expected_outputs = [
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1;
];

eta = 0.05;

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
unit_functions.epsilon = .1;

net = neural_network([rows(patterns), rows(expected_outputs)], unit_functions);

% Train the network
[net, train_memory] = net.train(net, patterns, expected_outputs, eta);
