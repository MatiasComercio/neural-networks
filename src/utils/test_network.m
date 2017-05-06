function [finished, test_memory] = test_network(net, patterns, expected_outputs, max_error)
   % Take patterns that haven't been used to train the network, and use the
   % network to aproximate their expected_output.
   % If the mean square of all aproximated oputputs is lesser than max_error,
   % then the network training is considered finished
  
   patterns_amount = columns(patterns);
   outputs = zeros(rows(expected_outputs), patterns_amount);
  
   % Use the network to solve each test pattern
   for i = 1:patterns_amount
       [output, memory] = net.solve(net, patterns(:, i));
       outputs(i) = output;
       test_memory(i).memory = memory;
   end
   
   % Find the error for the previously calculated patterns
   error = mean_square_error(expected_outputs, outputs);
   finished = error < max_error;
   
end

