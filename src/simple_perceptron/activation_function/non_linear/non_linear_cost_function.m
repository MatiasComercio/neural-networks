function global_error = non_linear_cost_function(expected_outputs, neural_outputs)
  all_differences = expected_outputs - neural_outputs;
  global_error = 0;
  for difference = all_differences
    global_error = global_error + difference^2;
  end
  global_error = global_error / 2;
end
