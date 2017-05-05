% Activation function
function ret = non_linear_tanh_g (h)
  beta = 1; % TODO: read from input file
  ret = tanh(beta .* h);
end
