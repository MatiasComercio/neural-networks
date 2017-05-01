% Delta function
function ret = non_linear_delta(expected_outputs, neural_outputs)
  % FIXME: uncomment this and delete below
  ret = (expected_outputs - neural_outputs) .* (1-neural_outputs^2);
%   ret = (expected_outputs - neural_outputs) .* neural_outputs(1-neural_outputs);
end

