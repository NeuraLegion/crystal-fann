module Crystal::Fann
  class Network
    property :nn

    def initialize(input : Int32, hidden : Array(Int32), output : Int32)
      @logger = Logger.new(STDOUT)
      layers = Array(UInt32).new
      layers << input.to_u32
      hidden.each do |l|
        layers << l.to_u32
      end
      layers << output.to_u32
      @output_size = output
      @input_size = input
      @nn = LibFANN.create_standard_array(layers.size, layers.to_unsafe)
    end

    def mse
      LibFANN.get_mse(@nn)
    end

    def close
      LibFANN.destroy(@nn)
    end

    def train_algorithem(algo : LibFANN::TrainEnum)
      LibFANN.set_training_algorithm(@nn, algo)
    end

    def set_hidden_layer_activation_func(func : LibFANN::ActivationfuncEnum)
      LibFANN.set_activation_function_hidden(@nn, func)
    end

    def set_output_layer_activation_func(func : LibFANN::ActivationfuncEnum)
      LibFANN.set_activation_function_output(@nn, func)
    end

    def train_single(input : Array(Float32), output : Array(Float32))
      raise "Input size not equal to input neurons" if input.size != @input_size
      raise "Output size not equal to output neurons" if output.size != @output_size
      LibFANN.train(@nn, input.to_unsafe, output.to_unsafe)
    end

    def train_array(input : Array(Array(Float32)), output : Array(Array(Float32)), opts = {:max_runs => 200, :desired_mse => 0.01_f32, :log_each => 1})
      raise "Input and Output arrays are not at the same size" if input.size != output.size
      runs = 0
      log_count = 0
      loop do
        input.each_with_index do |input_array, i|
          train_single(input_array, output[i])
        end
        # Check if we need to log MSE
        if log_count >= opts[:log_each]
          @logger.info("Run: #{runs}, MSE: #{mse}")
          STDOUT.flush
          log_count = 0
        else
          log_count += 1
        end
        # Check MSE and break if we reached goal
        break if mse <= opts[:desired_mse]
        # Check run count and break if needed
        runs += 1
        break if runs >= opts[:max_runs]
      end
    end

    def run(input : Array(Float32))
      result = LibFANN.run(@nn, input.to_unsafe)
      Slice.new(result, @output_size).to_a
    end

    def test
    end
  end
end
