% Delta function
function ret = delta(expected_outputs, neural_output)
  ret = expected_outputs - neural_output;
end

