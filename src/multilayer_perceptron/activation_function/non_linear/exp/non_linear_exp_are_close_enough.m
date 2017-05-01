function ret = non_linear_exp_are_close_enough(expected_output, neural_output)
  ret = abs(expected_output - neural_output) < .1;
end
