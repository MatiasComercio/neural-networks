% tanh derivative function
function ret = non_linear_tanh_g_derivative(g_output)
% Calculates the derivative tanh function based on the output of the tanh_g
% function
  config = get_config('non_linear_tanh_g');
  beta = config.beta;
  ret = beta .* (1 - g_output .^ 2);
end

