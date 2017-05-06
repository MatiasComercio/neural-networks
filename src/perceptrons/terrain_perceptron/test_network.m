function finished = test_network(net, patterns, expected_outputs)
   % Take patterns that haven't been used to train the network, and use the
   % network to aproximate their expected_output.
   % If the mean square of all aproximated oputputs is lesser than max_error,
   % then the network training is considered finished
   
   max_error = 5; % TODO: Chose an appropiate value
   patterns_amount = columns(patterns);
   outputs = zeros(1, patterns_amount);
   
   % Use the network to solve each test pattern
   for i = 1:patterns_amount
       [output, ~] = net.solve(net, patterns(:, i)); % ~ means all other return values are ignored
       outputs(i) = output;
   end
   
   % Find the error for the previously calculated patterns
   error = mean_square_error(expected_outputs, outputs);
   
   finished = error < max_error;
   
end

