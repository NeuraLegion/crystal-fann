# crystal-fann

Crystal bindings for the FANN C lib

## Installation

Add this to your application's `shard.yml`:

```yaml
dependencies:
  crystal-fann:
    github: bararchy/crystal-fann
```

## Usage

Look at the spec for most functions

```crystal
require "crystal-fann"
ann = Crystal::Fann::Network.new(2, [2], 1)
500.times do
  ann.train_single([1.0_f32, 0.1_f32], [0.5_f32])
end
result = ann.run([1.0_f32, 0.1_f32])
# Remmber to close the network when done to free allocated C mem
ann.close
```

```crystal
# Work on array of test data (one at a time -- not batch)
ann = Crystal::Fann::Network.new(2, [3], 1)
input = [[0.0_f32, 0.0_f32], [0.0_f32, 1.0_f32], [1.0_f32, 0.0_f32], [1.0_f32, 1.0_f32]]
output = [[0.0_f32], [1.0_f32], [1.0_f32], [0.0_f32]]
ann.train_algorithem(LibFANN::TrainEnum::TrainRprop)
ann.set_hidden_layer_activation_func(LibFANN::ActivationfuncEnum::Linear)
ann.set_output_layer_activation_func(LibFANN::ActivationfuncEnum::Linear)
ann.train_array(input, output, {:max_runs => 8000, :desired_mse => 0.001_f32, :log_each => 1000})
result = ann.run([1.0_f32, 1.0_f32])
ann.close
(result < [0.1]).should be_true
```

```crystal
# Work on array of test data (batch)
ann = Crystal::Fann::Network.new(2, [3], 1)
input = [[0.0_f32, 0.0_f32], [0.0_f32, 1.0_f32], [1.0_f32, 0.0_f32], [1.0_f32, 1.0_f32]]
output = [[0.0_f32], [1.0_f32], [1.0_f32], [0.0_f32]]
train_data = Crystal::Fann::TrainData.new(input, output)
data = train_data.train_data
ann.train_algorithem(LibFANN::TrainEnum::TrainRprop)
ann.set_hidden_layer_activation_func(LibFANN::ActivationfuncEnum::Linear)
ann.set_output_layer_activation_func(LibFANN::ActivationfuncEnum::Linear)
if data
  ann.train_batch(data, {:max_runs => 8000, :desired_mse => 0.001_f64, :log_each => 1000})
end
result = ann.run([1.0_f32, 1.0_f32])
ann.close
(result < [0.1]).should be_true
```

## Development
All C lib docs can be found here -> http://libfann.github.io/fann/docs/files/fann-h.html  

- [x] Add TrainData class  
- [x] Add network call method to train on train data  
- [ ] Add binding to the 'Parallel' binding to work on multi CPU at same time  
- [ ] Clean uneeded bindings in the LibFANN binding  
- [ ] Add specific Exceptions  
- [ ] Add binding and checks for lib errors  

I guess more stuff will be added once more people will use it.  

## Contributing

1. Fork it ( https://github.com/bararchy/crystal-fann/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

## Contributors

- [bararchy](https://github.com/bararchy) - creator, maintainer
- [libfann](https://github.com/libfann/fann) - c lib creators
