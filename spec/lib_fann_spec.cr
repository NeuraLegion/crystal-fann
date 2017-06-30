require "./spec_helper"

# TODO: Write tests
it "works" do
  u32_layers = Array(UInt32).new(4, 1_u32)
  fann = LibFANN.create_standard_array(u32_layers.size, u32_layers.to_unsafe)
  fann.is_a?(LibFANN::Fann*).should be_true
end
