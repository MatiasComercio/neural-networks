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
  -1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
];

eta = 0.05;
epsilon = .1;
alpha = 0.5;
gap = 100;

% Define which unit functions are going to be used
with_epsilon_are_close_enough = @(expected_output, neural_output) ...
    (epsilon_are_close_enough(expected_output, neural_output, epsilon));    
cost_function = @mean_square_error;

layers = create_all_non_linear_layers...
    ([rows(patterns), rows(expected_outputs)]);
net = neural_network(layers, with_epsilon_are_close_enough, cost_function);

% Train the network
[net, train_memory] = ...
    net.train(net, patterns, expected_outputs, eta, alpha, gap);
