require "./spec_helper"

describe Crystal::Fann do
  # TODO: Write tests

  it "initializes" do
    ann = Crystal::Fann::Network.new(2, [2, 2], 2)
    ann.nn.is_a?(LibFANN::Fann*).should be_true
  end

  it "free memory" do
    ann = Crystal::Fann::Network.new(2, [2, 2], 2)
    ann.close
    (ann.nn.is_a?(Pointer(Nil))).should be_true
  end
end
