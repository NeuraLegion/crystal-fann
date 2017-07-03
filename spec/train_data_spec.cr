require "./spec_helper"

describe Crystal::Fann::Network do
  it "initializes" do
    # initialize(input_array : Array(Array(Float32)), output_array : Array(Array(Float32)), max_input : Int32, max_output : Int32)
    input = [[0.0_f32, 0.0_f32], [0.0_f32, 1.0_f32], [1.0_f32, 0.0_f32], [1.0_f32, 1.0_f32]]
    output = [[0.0_f32], [1.0_f32], [1.0_f32], [0.0_f32]]
    train_data = Crystal::Fann::TrainData.new(input, output)
    (train_data.is_a?(Crystal::Fann::TrainData)).should be_true
  end
end
