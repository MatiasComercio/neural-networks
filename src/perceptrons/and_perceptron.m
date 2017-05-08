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
epsilon = .1;

% Define which unit functions are going to be used
unit_function = 'non_linear_tanh';
are_close_enough = str2func(strcat(unit_function, '_are_close_enough')); % TODO: migrate this outside the activation_functions folder
with_epsilon_are_close_enough = @(expected_output, neural_output) ...
    (are_close_enough(expected_output, neural_output, epsilon));
cost_function = @mean_square_error;

layers = create_all_non_linear_layers...
    ([rows(patterns), rows(expected_outputs)]);
net = neural_network(layers, with_epsilon_are_close_enough, cost_function);

% Train the network
[net, train_memory] = net.train(net, patterns, expected_outputs, eta);
