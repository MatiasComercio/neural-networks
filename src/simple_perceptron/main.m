% Load the root project's folder from the environmental variables
root_project_path = getenv('NEURAL_NETWORK_ROOT');
sp_path = fullfile(root_project_path, 'src', 'simple_perceptron');
% Add all src folders & subfolders to the current path
addpath(genpath(fullfile(sp_path, 'helper')));
addpath(genpath(fullfile(sp_path, 'perceptron')));
addpath(genpath(fullfile(sp_path, 'activation_function', 'unit_step')));

% Call the desired nerual network
or_perceptron;
weights

and_perceptron;
weights
