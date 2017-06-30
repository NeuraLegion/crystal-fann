require "./crystal-fann/*"

module Crystal::Fann
  class Network
    property :nn, :layers

    def initialize(input : Int32, hidden : Array(Int32), output : Int32)
      layers = Array(UInt32).new
      layers << input.to_u32
      hidden.each do |l|
        layers << l.to_u32
      end
      layers << output.to_u32
      @layers = layers
    end

    def new_network
      @nn = LibFANN.create_standard_array(layers.size, layers.to_unsafe)
    end

    def close
      @nn.destroy(self)
    end
  end
end
