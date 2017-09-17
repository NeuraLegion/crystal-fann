require "./spec_helper"

describe Crystal::Fann::Network do
  it "initializes" do
    # initialize(input_array : Array(Array(Float64)), output_array : Array(Array(Float64)), max_input : Int32, max_output : Int32)
    input = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    output = [[0.0], [1.0], [1.0], [0.0]]
    train_data = Crystal::Fann::TrainData.new(input, output)
    (train_data.is_a?(Crystal::Fann::TrainData)).should be_true
  end
end
