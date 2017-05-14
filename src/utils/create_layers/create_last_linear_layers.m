function layers = create_last_linear_layers(neurons_per_layer)
  layers = create_all_non_linear_layers(neurons_per_layer);
  layers_amount = length(neurons_per_layer);
  layers(layers_amount).g = @linear_identity_g;
  layers(layers_amount).g_derivative = @linear_identity_g_derivative;
  layers(layers_amount).neurons = neurons_per_layer(layers_amount);
end
