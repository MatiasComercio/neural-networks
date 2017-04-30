function ret = linear_are_close_enough(expected_output, neural_output)
  ret = sign(expected_output) == sign(neural_output);
end
