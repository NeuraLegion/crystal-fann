# crystal-fann

Crystal bindings for the FANN C lib

## Installation

Add this to your application's `shard.yml`:

```yaml
dependencies:
  crystal-fann:
    github: NeuraLegion/crystal-fann
```

## Usage

Look at the spec for most functions

```crystal
require "crystal-fann"
ann = Crystal::Fann::Network::Standard.new(2, [2], 1)
500.times do
  ann.train_single([1.0, 0.1], [0.5])
end
result = ann.run([1.0, 0.1])
# Remmber to close the network when done to free allocated C mem
ann.close
```

```crystal
# Work on array of test data (batch)
ann = Crystal::Fann::Network::Standard.new(2, [3], 1)
input = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
output = [[0.0], [1.0], [1.0], [0.0]]
train_data = Crystal::Fann::TrainData.new(input, output)
data = train_data.train_data
ann.train_algorithem(LibFANN::TrainEnum::TrainRprop)
ann.set_hidden_layer_activation_func(LibFANN::ActivationfuncEnum::Linear)
ann.set_output_layer_activation_func(LibFANN::ActivationfuncEnum::Linear)
if data
  ann.train_batch(data, {:max_runs => 8000, :desired_mse => 0.001_f64, :log_each => 1000})
end
result = ann.run([1.0, 1.0])
ann.close
(result < [0.1]).should be_true
```

```crystal
# Work on array of test data using the Cascade2 algorithm (no hidden layers, net will build it alone)
ann = Crystal::Fann::Network::Cascade.new(2, 1)
input = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
output = [[0.0], [1.0], [1.0], [0.0]]
train_data = Crystal::Fann::TrainData.new(input, output)
data = train_data.train_data
ann.train_algorithem(LibFANN::TrainEnum::TrainRprop)
ann.set_hidden_layer_activation_func(LibFANN::ActivationfuncEnum::Linear)
ann.set_output_layer_activation_func(LibFANN::ActivationfuncEnum::Linear)
if data
  ann.train_batch(data, {:max_neurons => 500, :desired_mse => 0.1_f64, :log_each => 10})
end
result = ann.run([1.0, 1.0])
ann.close
(result < [0.1]).should be_true
```

## Development
All C lib docs can be found here -> http://libfann.github.io/fann/docs/files/fann-h.html  

- [x] Add TrainData class  
- [x] Add network call method to train on train data  
- [x] Add binding to the 'Parallel' binding to work on multi CPU at same time  
- [ ] Clean uneeded bindings in the LibFANN binding  
- [ ] Add specific Exceptions  
- [ ] Add binding and checks for lib errors  

I guess more stuff will be added once more people will use it.  

## Contributing

1. Fork it ( https://github.com/NeuraLegion/crystal-fann/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

## Contributors

- [bararchy](https://github.com/bararchy) - creator, maintainer
- [libfann](https://github.com/libfann/fann) - c lib creators
