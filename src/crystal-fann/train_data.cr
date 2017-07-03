# struct TrainData
#   errno_f : ErrnoEnum
#   error_log : File*
#   errstr : LibC::Char*
#   num_data : LibC::UInt
#   num_input : LibC::UInt
#   num_output : LibC::UInt
#   input : Type**
#   output : Type**
# end
module Crystal::Fann
  class TrainData
    property :data_struct

    def initialize(input_array : Array(Array(Float32)), output_array : Array(Array(Float32)))
      @input_array = input_array
      @output_array = output_array
      @n_i = (@input_array.map &.to_unsafe).as Array(Pointer(Float32))
      @n_o = (@output_array.map &.to_unsafe).as Array(Pointer(Float32))
      @p_input_array = (@n_i.to_unsafe).as Pointer(Pointer(Float32))
      @p_output_array = (@n_o.to_unsafe).as Pointer(Pointer(Float32))
      set_train_struct
    end

    def set_train_struct
      data_struct = LibFANN::TrainData.new.as LibFANN::TrainData
      if data_struct.input = @p_input_array
        data_struct.output = @p_output_array
        data_struct.num_input = @input_array.first.size
        data_struct.num_output = @output_array.first.size
        data_struct.num_data = @input_array.size
      end
      @data_struct = data_struct.as LibFANN::TrainData
    end

    def train_data : Pointer(LibFANN::TrainData)
      ptr = pointerof(@data_struct).as Pointer(LibFANN::TrainData)
      unless ptr.is_a?(Pointer(LibFANN::TrainData))
        raise "Can't set train data #{ptr}, #{ptr.class}"
      end
      ptr
    end
  end
end
