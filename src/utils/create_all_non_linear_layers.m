function layers = create_all_non_linear_layers(neurons_per_layer)
  layers(length(neurons_per_layer)).neurons = 0;
  for i = 1:length(neurons_per_layer)
    layers(i).g = @non_linear_tanh_g;
    layers(i).g_derivative = @non_linear_tanh_g_derivative_improved;
    layers(i).neurons = neurons_per_layer(i);
  end
end

