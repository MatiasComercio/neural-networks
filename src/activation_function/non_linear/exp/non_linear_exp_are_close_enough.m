function ret = non_linear_exp_are_close_enough(expected_output, neural_output, epsilon)
  ret = abs(expected_output - neural_output) < epsilon;
end
