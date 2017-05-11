% tanh derivative function
function ret = non_linear_exp_g_derivative(g_output)
% Calculates the derivative tanh function based on the output of the tanh_g
% function
  config = get_config('non_linear_exp_g');
  beta = config.beta;
  ret = -2 * beta .* g_output .* (1 - g_output);
end

