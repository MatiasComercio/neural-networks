function layers = create_layers( neurons_per_layer, all_g, all_g_derivate, ...
    last_g, last_g_derivate )
    layers_amount = length(neurons_per_layer);
    
    layers(layers_amount).g = last_g;
    layers(layers_amount).g_derivative = last_g_derivate;
	layers(layers_amount).neurons = neurons_per_layer(layers_amount);
    
	for i = 1:layers_amount-1
        layers(i).g = all_g;
        layers(i).g_derivative = all_g_derivate;
        layers(i).neurons = neurons_per_layer(i);
	end
end
