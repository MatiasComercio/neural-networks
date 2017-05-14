function layers = create_layers( neurons_per_layer, hidden_g, ...
    hidden_g_derivative, last_g, last_g_derivative )
    layers_amount = length(neurons_per_layer);
    
    layers(layers_amount).g = last_g;
    layers(layers_amount).g_derivative = last_g_derivative;
	layers(layers_amount).neurons = neurons_per_layer(layers_amount);
    
	for i = 1:layers_amount-1
        layers(i).g = hidden_g;
        layers(i).g_derivative = hidden_g_derivative;
        layers(i).neurons = neurons_per_layer(i);
	end
end
