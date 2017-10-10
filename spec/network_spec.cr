require "./spec_helper"

describe Fann::Network do
  it "initializes standard network" do
    ann = Fann::Network::Standard.new(2, [2, 2], 2)
    ann.nn.is_a?(LibFANN::Fann*).should be_true
  end

  it "initializes cascade network" do
    ann = Fann::Network::Cascade.new(2, 2)
    ann.nn.is_a?(LibFANN::Fann*).should be_true
  end

  it "free memory" do
    ann = Fann::Network::Standard.new(2, [2, 2], 2)
    ann.close
  end

  it "trains on single data" do
    ann = Fann::Network::Standard.new(2, [2, 2], 1)
    ann.randomize_weights(0.0, 1.0)
    3000.times do
      ann.train_single([1.0, 0.0], [0.5])
    end
    (ann.mse < 0.01).should be_true
    ann.close
  end

  it "Shows MSE" do
    ann = Fann::Network::Standard.new(2, [2, 2], 2)
    ann.mse.is_a?(LibC::Double).should be_true
    ann.close
  end

  it "trains and evaluate single data" do
    ann = Fann::Network::Standard.new(2, [2], 1)
    ann.randomize_weights(0.0, 1.0)
    3000.times do
      ann.train_single([1.0, 0.0], [0.5])
    end
    result = ann.run([1.0, 0.0])
    ann.close
    (result < [0.55] && result > [0.45]).should be_true
  end

  it "train on batch" do
    ann = Fann::Network::Standard.new(2, [3], 1)
    input = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    output = [[0.0], [1.0], [1.0], [0.0]]
    train_data = Fann::TrainData.new(input, output)
    data = train_data.train_data
    # ann.train_algorithem(LibFANN::TrainEnum::TrainSarprop)
    ann.randomize_weights(0.0, 1.0)
    # ann.set_hidden_layer_activation_func(LibFANN::ActivationfuncEnum::LeakyRelu)
    # ann.set_output_layer_activation_func(LibFANN::ActivationfuncEnum::LeakyRelu)
    if data
      ann.train_batch(data, {:max_runs => 8000, :desired_mse => 0.001, :log_each => 1000})
    end
    result = ann.run([1.0, 1.0])
    ann.close
    (result < [0.15]).should be_true
  end

  it "train on cascade" do
    ann = Fann::Network::Cascade.new(2, 1)
    input = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    output = [[0.0], [1.0], [1.0], [0.0]]
    train_data = Fann::TrainData.new(input, output)
    data = train_data.train_data
    ann.train_algorithm(LibFANN::TrainEnum::TrainRprop)
    ann.randomize_weights(0.0, 1.0)
    if data
      ann.train_batch(data, {:max_neurons => 500, :desired_mse => 0.001, :log_each => 10})
    end
    result = ann.run([1.0, 1.0])
    ann.close
    (result < [0.1]).should be_true
  end

  context "saving current network" do
    it "saves standard networks" do
      tempfile = Tempfile.new("foo")
      File.size(tempfile.path).should eq 0
      ann = Fann::Network::Standard.new(2, [2], 1)
      ann.save(tempfile.path)
      (File.size(tempfile.path) > 0).should be_true
    end

    it "saves cascade networks" do
      tempfile = Tempfile.new("bar")
      File.size(tempfile.path).should eq 0
      ann = Fann::Network::Cascade.new(2, 1)
      ann.save(tempfile.path)
      (File.size(tempfile.path) > 0).should be_true
    end
  end

  context "loading a configuration file" do
    it "loads standard networks" do
      input = 2
      output = 1
      tempfile = Tempfile.new("standard")
      original = Fann::Network::Standard.new(input, [2], output)
      original.save(tempfile.path)
      loaded = Fann::Network::Standard.new(tempfile.path)
      loaded.input_size.should eq input
      loaded.output_size.should eq output
    end

    it "loads cascade networks" do
      input = 2
      output = 1
      tempfile = Tempfile.new("cascade")
      original = Fann::Network::Cascade.new(input, output)
      original.save(tempfile.path)
      loaded = Fann::Network::Cascade.new(tempfile.path)
      loaded.input_size.should eq input
      loaded.output_size.should eq output
    end
  end
end
