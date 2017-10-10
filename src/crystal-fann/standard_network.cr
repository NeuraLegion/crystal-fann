module Fann
  module Network
    class Standard
      property :nn
      getter :input_size
      getter :output_size

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

      def initialize(path : String)
        @logger = Logger.new(STDOUT)
        @nn = LibFANN.create_from_file(path)
        @input_size = LibFANN.get_num_input(@nn)
        @output_size = LibFANN.get_num_output(@nn)
      end

      def mse
        LibFANN.get_mse(@nn)
      end

      def close
        LibFANN.destroy(@nn)
      end

      def train_algorithm(algo : LibFANN::TrainEnum)
        LibFANN.set_training_algorithm(@nn, algo)
      end

      def set_hidden_layer_activation_func(func : LibFANN::ActivationfuncEnum)
        LibFANN.set_activation_function_hidden(@nn, func)
      end

      def set_output_layer_activation_func(func : LibFANN::ActivationfuncEnum)
        LibFANN.set_activation_function_output(@nn, func)
      end

      def randomize_weights(min : Float64, max : Float64)
        # randomize_weights = fann_randomize_weights(ann : Fann*, min_weight : Type, max_weight : Type)
        LibFANN.randomize_weights(@nn, min, max)
      end

      def train_single(input : Array(Float64), output : Array(Float64))
        raise "Input size not equal to input neurons" if input.size != @input_size
        raise "Output size not equal to output neurons" if output.size != @output_size
        LibFANN.train(@nn, input.to_unsafe, output.to_unsafe)
      end

      def train_batch(train_data : Pointer(LibFANN::TrainData), opts = {:max_runs => 200, :desired_mse => 0.01_f64, :log_each => 1})
        # train_on_data = fann_train_on_data(ann : Fann*, data : TrainData*, max_epochs : LibC::UInt, epochs_between_reports : LibC::UInt, desired_error : LibC::Double)
        LibFANN.train_on_data(@nn, train_data, opts[:max_runs], opts[:log_each], opts[:desired_mse].to_f32)
      end

      def train_batch_multicore(train_data : Pointer(LibFANN::TrainData), threads : Int32, opts = {:max_runs => 200, :desired_mse => 0.01_f64, :log_each => 1})
        # fun train_epoch_irpropm_parallel = fann_train_epoch_irpropm_parallel(ann : Fann*, data : TrainData*, threadnumb : LibC::UInt) : LibC::Double
        runs = 0
        log_count = 0
        loop do
          LibFANN.train_epoch_irpropm_parallel(@nn, train_data, threads)
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

      def run(input : Array(Float64))
        result = LibFANN.run(@nn, input.to_unsafe)
        Slice.new(result, @output_size).to_a
      end

      def save(path : String) : Int32
        LibFANN.save(@nn, path)
      end
    end
  end
end
