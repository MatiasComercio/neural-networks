patterns = [
  -1;
  1;
];

expected_outputs = 1;

eta = 0.5;
epsilon = 0.1;
alpha = 0.9;
gap = 1;

% Define which unit functions are going to be used
with_epsilon_are_close_enough = @(expected_output, neural_output) ...
    (epsilon_are_close_enough(expected_output, neural_output, epsilon));
cost_function = @mean_square_error;
eval_gap = @evaluate_gap;

layers = create_all_non_linear_layers...
    ([rows(patterns), 2, rows(expected_outputs)]);
net = neural_network(layers, with_epsilon_are_close_enough, cost_function);
net.layers(1).weights = [-.35, .15, .20; -.35, .25, .30];
net.layers(2).weights = [-.60, .40, .45];

% Train the network
[net, train_memory] = net.train(net, patterns, expected_outputs, ...
    eta, alpha, gap, eval_gap);

w_1_new = train_memory(2).layers(1).weights;
w_2_new = train_memory(2).layers(2).weights;
eta_2 = train_memory(2).eta;

expected_w_1_new = [ -0.424478186335586, 0.075521813664414, 0.274478186335586; -0.433452805104848, 0.166547194895152, 0.383452805104848 ];
expected_w_2_new = [ -0.794609484237469, 0.477381346655652, 0.527792930394303 ];
expected_eta = eta + 0.001; % Error decremented in the first gap iteration

epsilon = 1e-4;
delta_w_1 = ones(2, 3) .* epsilon;
delta_w_2 = ones(1, 3) .* epsilon;

assert(all(all(abs(w_1_new - expected_w_1_new) < delta_w_1)), ...
    '[FAIL] - w_1_new')
assert(all(all(abs(w_2_new - expected_w_2_new) < delta_w_2)), ...
    '[FAIL] - w_2_new')
assert(abs(eta_2 - expected_eta) < epsilon, '[FAIL] - eta')
