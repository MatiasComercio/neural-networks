function out = get_config( input_string )
    out = [];

    switch lower(input_string)
        case 'non_linear_exp_g'
            out.beta = 1/2;
        case 'non_linear_tanh_g'
            out.beta = 1;
        case 'terrain_perceptron'
            out.sample.size = 441;
            out.epsilon = 0.1;
            out.gap.size = 1;
            out.gap.eval = @evaluate_gap;
            out.eta = 0.1;
            out.alpha = 0.9;
            out.layers.neurons = [2, 10, 10, 5, 1];
            % Possible: [non_linear_tanh_g, linear_identity_g]
            out.layers.hidden.g = @non_linear_tanh_g;
            % Possible: [non_linear_tanh_g_derivative_improved, linear_identity_g_derivative]
            out.layers.hidden.g_derivate = @non_linear_tanh_g_derivative_improved;
            % Possible: [non_linear_tanh_g, linear_identity_g]
            out.layers.last.g = @non_linear_tanh_g;
            % Possible: [non_linear_tanh_g_derivative_improved, linear_identity_g_derivative]
            out.layers.last.g_derivate = @non_linear_tanh_g_derivative_improved;
        otherwise
            throw(MException('ITBA-SIA:noSuchConfig', '%s not found', ...
                upper(input_string)));
    end
end
