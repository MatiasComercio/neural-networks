function ret = unit_step_are_close_enough(expected_output, neural_output)
  ret = expected_output - neural_output == 0;
end
