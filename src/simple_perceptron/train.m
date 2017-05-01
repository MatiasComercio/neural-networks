function [weights, global_errors] = train(patterns, expected_outputs, ...
    g, delta, are_close_enough, cost_function)
% weights: row vector
% pattern: column vector
% patterns: matrix with each column being a pattern
% extended_patterns: patterns with the first row being -1 (constant bias factor)
% eta: learning rate
% expected_outputs: row vector containing the expected output of patterns(i) at 
%   position i.
% neural_output: row vector containint the resulting output of patterns(i) at
%   position i, that was obtained using the current `g` function and the current
%   weigths vector 

  % Read eta
  eta = 0.05;
  
  % Build the extended patterns
  bias_constant_factors = ones(1, columns(patterns)) .* -1;
  extended_patterns = [ 
    bias_constant_factors;
    patterns;
  ];

  % Create the weights row vector with values between [-0.5; 0.5]
  weights = initialize_weights(rows(extended_patterns));
  % Train until expected outputs match the neural outputs
  finished = false;
  era_counter = 0;
  global_errors = [];
  while ~finished
    print_var('------ ERA ------', era_counter);
    % Train one complete era
    [neural_outputs, weights] = ...
        era(weights, extended_patterns, expected_outputs, eta, g, delta);
    global_errors = [global_errors; ...
        feval(cost_function, expected_outputs, neural_outputs)];
    % Print current results
    print_var('neural_outputs', neural_outputs);
    print_var('weights', weights);
    % Determine whether the outputs match
    finished = do_outputs_match(expected_outputs, neural_outputs, ...
        are_close_enough);
    era_counter = era_counter + 1;
  end
end

function [neural_outputs, weights] = ...
        era(weights, extended_patterns, expected_outputs, eta, g, delta) 
  % Create a permutation to get a different sample patterns
  % Recall each pattern is a column vector, so, the number of columns of
  %   the extended_patterns matrix is the number of total patterns
  sample_patterns_indexes = randperm(columns(extended_patterns));
  % Initialize the output row vector
  neural_outputs = zeros(1, columns(expected_outputs));
  % Train with each pattern in the specified random order
  for i = sample_patterns_indexes
    % Get the current sample pattern
    pattern = extended_patterns(:,i);
    % Print all parameters
    print_var('** New Training Pattern **', pattern);
    print_var('weights', weights);
    % Get the current pattern's output with the current weights
    neural_outputs(i) = feval(g, weights * pattern);
    % Fix weights for the current pattern
    weights_fixes = eta * feval(delta, expected_outputs(i), ...
        neural_outputs(i)) .* pattern.';
    weights = weights + weights_fixes;
    % Print neural output and weights fixes
    print_var('neural_output', neural_outputs(i));
    print_var('weights_fixes', weights_fixes);
  end
end

function ret = initialize_weights(length)
  ret = random_row_in_range(-0.5, 0.5, length);
end


function ret = random_row_in_range(min, max, length)
  % min: inclusive
  % max: inclusive
  % length: row vector length
  ret = (max-min).*rand(1, length) + min;
end

function is_expected = do_outputs_match(expected_outputs, ...
    neural_outputs, are_close_enough)
  is_expected = true;
  for i = 1:length(expected_outputs)
    is_expected = is_expected && ...
        feval(are_close_enough, expected_outputs(i), neural_outputs(i));
    if ~ is_expected
      return;
    end
  end
end

function ret = columns(m)
  ret = size(m, 2);
end

function ret = rows(m)
  ret = size(m, 1);
end

