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

% Define which unit functions are going to be used
unit_function = 'non_linear';
g = strcat(unit_function, '_g');
delta = strcat(unit_function, '_delta');
are_close_enough = strcat(unit_function, '_are_close_enough');
cost_function = strcat(unit_function, '_cost_function');

% Train the network
[weights, global_errors] = train(patterns, expected_outputs, g, delta, ...
    are_close_enough, cost_function);

% Print solution
print_var('Solution weights', weights);
