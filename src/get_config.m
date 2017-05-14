function out = get_config( input_string )
    out = [];

    switch lower(input_string)
        case 'non_linear_exp_g'
            out.beta = 1/2;
        case 'non_linear_tanh_g'
            out.beta = 1;
        case 'evaluate_gap'
            out.a = 0.001;
            out.b = 0.1;
            out.k = 3;
        case 'terrain_perceptron'
            out.filename = 'terrain_perceptron_net.mat';
            [out.patterns, out.expected_outputs] = terrain_data();
            out.data_size = columns(out.patterns);
            out.epsilon = 0.1;
            out.gap.size = 1;
            out.gap.eval = @evaluate_gap;
            out.eta = 0.05;
            out.alpha = 0.9;
            out.layers.neurons = [rows(out.patterns), 10, 5, 5, 3, rows(out.expected_outputs)];
            % Possible: non_linear_tanh_g
            %           non_linear_exp_g
            %           linear_identity_g
            out.layers.hidden.g = @non_linear_tanh_g;
            % Possible: non_linear_tanh_g_derivative
            %           non_linear_tanh_g_derivative_improved
            %           non_linear_exp_g_derivative
            %           non_linear_exp_g_derivative_improved
            %           linear_identity_g_derivative
            out.layers.hidden.g_derivative = @non_linear_tanh_g_derivative_improved;
            % Possible: non_linear_tanh_g
            %           non_linear_exp_g
            %           linear_identity_g
            out.layers.last.g = @linear_identity_g;
            % Possible: non_linear_tanh_g_derivative
            %           non_linear_tanh_g_derivative_improved
            %           non_linear_exp_g_derivative
            %           non_linear_exp_g_derivative_improved
            %           linear_identity_g_derivative
            out.layers.last.g_derivative = @linear_identity_g_derivative;
        otherwise
            error('%s config not found', upper(input_string));
    end
end
