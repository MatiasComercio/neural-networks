# Neural Networks
TP2 for Artificial Intelligence Systems course from I.T.B.A.

## Perceptrons

The project has included 6 perceptrons:

- Currently working:
  - TERRAIN PERCEPTRON
- Currently not working (_used in development stage only_):
  - AND, OR, PARITY, SYMMETRY, XOR PERCEPTRONS

## Pre-requisites
Add the `src` folder & subfolders to the Matlab/Octave path.

## Configuration

Change the `src/get_config.m` file with desired parameters.

### Change non linear activation function's **beta**

```
case 'non_linear_exp_g'
    out.beta = 1/2;
case 'non_linear_tanh_g'
    out.beta = 1;'
```

### Change adaptive learning rate parameters

Where `a` and `b` are the adaptive constants and `k` refers to the number of continuous good epochs that have to occur in order to increment the learning rate.

```
case 'evaluate_gap'
    out.a = 0.001;
    out.b = 0.1;
    out.k = 3;
```

### Change when to plot while training

```
case 'neural_network'
    out.net_save_period = 50;
    out.error_bars_plot_period = 15;
```

### Terrain perceptron

#### Change the filename where the trained network will be saved

```
case 'terrain_perceptron'
    out.filename = 'terrain_perceptron_net.mat';
    ...
```

#### Change the perceptron's data

Where `data_size` is the sample size to take from `patterns` and `expected_outputs`. `terrain_data()` function returns the provided data for the perceptron.

```
case 'terrain_perceptron'
    ...
    [out.patterns, out.expected_outputs] = terrain_data();
    out.data_size = columns(out.patterns);
    ...
```

#### Change the perceptron's parameters

- `epsilon` is the error difference that has to be reached in order to stop training when comparing the network's outputs with the desired ones for the training patterns.
- `alpha` is the momentum parameter.
    - Disable momentum: set `alpha = 0`
- `gap.size` is the number of epochs that need to have passed in order to test if a good epoch was reached.
  - **IMPORTANT**: Currently unavailable option: You should always use `gap.size = 1`
- `gap.eval` it's the adaptive learning rate function. Possible values:
    - Use adaptive learning rate: `@dont_evaluate_gap`
    - Disable adaptive learning rate: `@evaluate_gap`

```
 case 'terrain_perceptron'
    ...
    out.epsilon = 0.1;
    out.eta = 0.05;
    out.alpha = 0.9;
    out.gap.size = 1;
    out.gap.eval = @dont_evaluate_gap;
    ...
```

#### Change the perceptron's architecture

- `layers.layers.neurons` is an array where each element represents a layer and sets the number of neurons for that layer.
- `layers.hidden.g` is the activation function for all the hidden layers and `layers.hidden.g_derivative ` its derivative.
- `layers.last.g` is the activation function for the last layer and `layers.last.g_derivative` its derivative.
- `layers.*.g` possible values:
    - `non_linear_tanh_g`
    - `non_linear_exp_g`
    - `linear_identity_g`
- `layers.*.g_derivative` possible values:
    - `non_linear_tanh_g_derivative`
    - `non_linear_tanh_g_derivative_improved`
    - `non_linear_exp_g_derivative`
    - `non_linear_exp_g_derivative_improved`
    - `linear_identity_g_derivative`

```
case 'terrain_perceptron'
    ...
    out.layers.neurons = [rows(out.patterns), 10, 5, 5, rows(out.expected_outputs)];
    out.layers.hidden.g = @non_linear_tanh_g;
    out.layers.hidden.g_derivative = @non_linear_tanh_g_derivative_improved;
    out.layers.last.g = @linear_identity_g;
    out.layers.last.g_derivative = @linear_identity_g_derivative;
```

## Usage

Run the desired perceptron's script from `src/perceptrons`.

### Terrain Perceptron

Run `src/perceptrons/tarrain_perceptron/terrain_perceptron.m` script after changing the perceptron's configuration with desired parameters.

The script will save a file with the name specified in the configuration with the trained network data. You can use this trained network to solve other data without learning by running the `src/perceptrons/tarrain_perceptron/trained_terrain_perceptron.m` script after changing it's content with desired parameters.

#### Example
From the Matlab/Octave command line, and after adding the `src` folder & subfolders to the Matlab/Octave path, execute

    terrain_perceptron

Then, you can do

    trained_terrain_perceptron

## Authors
This project is written and maintained by

- [Matías Nicolás Comercio Vázquez](https://github.com/MatiasComercio)
- [Gonzalo Ibars Ingman](https://github.com/gibarsin)
- [Matías Mercado](https://github.com/MatiasMercado)
- [Juan Moreno](https://github.com/jpmrno)

## License
    MIT License

    Copyright (c) 2017
      - Matías Nicolás Comercio Vázquez <mcomerciovazquez@gmail.com>
      - Gonzalo Ibars Ingman <gibarsin@itba.edu.ar>
      - Matías Mercado <mmercado@itba.edu.ar>
      - Juan Moreno <jpmrno@itba.edu.ar>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
