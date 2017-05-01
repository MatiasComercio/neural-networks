function ret = mean_square_error(expected_outputs, outputs)
  ret = .5 * sum(sum((expected_outputs - outputs).^2));
end

