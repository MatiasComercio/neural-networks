% Activation function
function ret = non_linear_exp_g (h)
  beta = 1/2; % TODO: read from input file
  ret = (1 + exp(-2 * beta .* h)) .^ -1;
end
