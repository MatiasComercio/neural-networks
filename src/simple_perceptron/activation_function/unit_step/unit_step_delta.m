% Delta function
function ret = step_delta(expected_outputs, neural_output)
  ret = expected_outputs - neural_output;
end

