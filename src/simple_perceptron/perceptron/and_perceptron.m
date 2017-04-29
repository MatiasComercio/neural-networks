% Read input patterns
patterns = [
  1, 1,-1,-1;
  1,-1, 1,-1;
];

% Read expected outputs
expected_outputs = [
  1, -1, -1, -1;
];

% Train the network
weights = train(patterns, expected_outputs);

% Print solution
print_var('Solution weights', weights);
