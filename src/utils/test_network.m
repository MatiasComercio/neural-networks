function [finished, test_memory] = test_network(net, patterns, expected_outputs, max_error)
   % Take patterns that haven't been used to train the network, and use the
   % network to aproximate their expected_output.
   % If the mean square of all aproximated oputputs is lesser than max_error,
   % then the network training is considered finished
  
   [outputs, test_memory.solve_memories] = solve_patterns(net, patterns);
   
   % Find the error for the previously calculated patterns
   test_memory.error = mean_square_error(expected_outputs, outputs);
   finished = test_memory.error < max_error;
end
