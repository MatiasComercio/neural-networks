% Activation function
function ret = non_linear_exp_g (h)
  config = get_config('non_linear_exp_g');
  beta = config.beta;
  ret = (1 + exp(-2 * beta .* h)) .^ -1;
end
