% Activation function
function ret = non_linear_tanh_g (h)
  config = get_config('non_linear_tanh_g');
  beta = config.beta;
  ret = tanh(beta .* h);
end
