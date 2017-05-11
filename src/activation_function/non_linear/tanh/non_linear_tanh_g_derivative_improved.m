% tanh derivative function
function ret = non_linear_tanh_g_derivative_improved(g_output)
% Calculates the derivative tanh function based on the output of the tanh_g
% function
  beta = 1; % TODO: read from input file
  ret = beta .* (1 - g_output .^ 2) + .1;
end

