require "./spec_helper"

describe Crystal::Fann::Network do
  it "initializes" do
    ann = Crystal::Fann::Network.new(2, [2, 2], 2)
    ann.nn.is_a?(LibFANN::Fann*).should be_true
  end

  it "free memory" do
    ann = Crystal::Fann::Network.new(2, [2, 2], 2)
    ann.close
  end

  it "trains on single data" do
    ann = Crystal::Fann::Network.new(2, [2, 2], 1)
    500.times do
      ann.train_single([1.0_f32, 0.0_f32], [0.5_f32])
    end
    (ann.mse < 0.01).should be_true
    ann.close
  end

  it "Shows MSE" do
    ann = Crystal::Fann::Network.new(2, [2, 2], 2)
    ann.mse.is_a?(LibC::Double).should be_true
    ann.close
  end

  it "trains and evaluate single data" do
    ann = Crystal::Fann::Network.new(2, [2], 1)
    500.times do
      ann.train_single([1.0_f32, 0.1_f32], [0.5_f32])
    end
    result = ann.run([1.0_f32, 0.1_f32])
    ann.close
    result.should eq([0.5])
  end

  it "trains and evaluate array data" do
    ann = Crystal::Fann::Network.new(2, [2], 1)
    input = [[1.0_f32, 0.1_f32], [0.4_f32, 0.2_f32], [0.3_f32, 0.1_f32]]
    output = [[0.5_f32], [0.3_f32], [0.2_f32]]
    ann.train_array(input, output, {:max_runs => 8000, :desired_mse => 0.001_f64, :log_each => 1000})
    result = ann.run([0.4_f32, 0.2_f32])
    ann.close
    (result.flatten < [0.4] && result.flatten > [0.2]).should be_true
  end

  it "solves XOR problem" do
    ann = Crystal::Fann::Network.new(2, [3], 1)
    input = [[0.0_f32, 0.0_f32], [0.0_f32, 1.0_f32], [1.0_f32, 0.0_f32], [1.0_f32, 1.0_f32]]
    output = [[0.0_f32], [1.0_f32], [1.0_f32], [0.0_f32]]
    ann.train_array(input, output, {:max_runs => 8000, :desired_mse => 0.001_f64, :log_each => 1000})
    result = ann.run([1.0_f32, 1.0_f32])
    ann.close
    (result < [0.1]).should be_true
  end

  it "set_activation_functions and algorithems" do
    ann = Crystal::Fann::Network.new(2, [3], 1)
    input = [[0.0_f32, 0.0_f32], [0.0_f32, 1.0_f32], [1.0_f32, 0.0_f32], [1.0_f32, 1.0_f32]]
    output = [[0.0_f32], [1.0_f32], [1.0_f32], [0.0_f32]]
    ann.train_algorithem(LibFANN::TrainEnum::TrainRprop)
    ann.set_hidden_layer_activation_func(LibFANN::ActivationfuncEnum::Linear)
    ann.set_output_layer_activation_func(LibFANN::ActivationfuncEnum::Linear)
    ann.train_array(input, output, {:max_runs => 8000, :desired_mse => 0.001_f64, :log_each => 1000})
    result = ann.run([1.0_f32, 1.0_f32])
    ann.close
    (result < [0.1]).should be_true
  end

  it "train on batch" do
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
  end

  it "train on cascade" do
    ann = Crystal::Fann::Network.new(2, [3], 1)
    input = [[0.0_f32, 0.0_f32], [0.0_f32, 1.0_f32], [1.0_f32, 0.0_f32], [1.0_f32, 1.0_f32]]
    output = [[0.0_f32], [1.0_f32], [1.0_f32], [0.0_f32]]
    train_data = Crystal::Fann::TrainData.new(input, output)
    data = train_data.train_data
    ann.train_algorithem(LibFANN::TrainEnum::TrainRprop)
    ann.set_hidden_layer_activation_func(LibFANN::ActivationfuncEnum::Linear)
    ann.set_output_layer_activation_func(LibFANN::ActivationfuncEnum::Linear)
    if data
      ann.train_batch_cascade(data, {:max_neurons => 500, :desired_mse => 0.1_f64, :log_each => 10})
    end
    result = ann.run([1.0_f32, 1.0_f32])
    ann.close
    (result < [0.1]).should be_true
  end

  it "train on multiple cores" do
    ann = Crystal::Fann::Network.new(2, [3], 1)
    input = [[0.0_f32, 0.0_f32], [0.0_f32, 1.0_f32], [1.0_f32, 0.0_f32], [1.0_f32, 1.0_f32]]
    output = [[0.0_f32], [1.0_f32], [1.0_f32], [0.0_f32]]
    train_data = Crystal::Fann::TrainData.new(input, output)
    data = train_data.train_data
    ann.train_algorithem(LibFANN::TrainEnum::TrainRprop)
    ann.set_hidden_layer_activation_func(LibFANN::ActivationfuncEnum::Linear)
    ann.set_output_layer_activation_func(LibFANN::ActivationfuncEnum::Linear)
    if data
      5000.times do
        ann.train_batch_multicore(data, 4, {:max_runs => 8000, :desired_mse => 0.001_f64, :log_each => 1000})
      end
    end
    result = ann.run([1.0_f32, 1.0_f32])
    ann.close
    (result < [0.1]).should be_true
  end
end
