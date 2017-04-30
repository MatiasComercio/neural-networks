% Delta function
function ret = non_linear_delta(expected_output, neural_output)
  ret = (expected_output - neural_output) * (1-neural_output^2);
end

