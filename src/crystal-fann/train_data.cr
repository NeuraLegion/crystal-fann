module Crystal::Fann
  class TrainData
    property :data_struct, :p_input_array, :p_output_array

    def initialize(input_array : Array(Array(Float32)), output_array : Array(Array(Float32)))
      # Save so GC wont collect
      @input_array = input_array
      @output_array = output_array
      # Map as an Array of Pointer(Float32)
      @n_i = (@input_array.map &.to_unsafe).as Array(Pointer(Float32))
      @n_o = (@output_array.map &.to_unsafe).as Array(Pointer(Float32))
      # Get the Pointer of Pointer(Float32)
      @p_input_array = (@n_i.to_unsafe).as Pointer(Pointer(Float32))
      @p_output_array = (@n_o.to_unsafe).as Pointer(Pointer(Float32))
      # Build struct
      set_train_struct
    end

    def set_train_struct
      # fun create_train_pointer_array = fann_create_train_pointer_array(num_data : LibC::UInt, num_input : LibC::UInt, input : Type**, num_output : LibC::UInt, output : Type**) : TrainData*
      data_struct = LibFANN.create_train_pointer_array(
        @input_array.size,
        @input_array.sample.size,
        @p_input_array,
        @output_array.sample.size,
        @p_output_array
      )
      @data_struct = data_struct.as Pointer(LibFANN::TrainData)
    end

    def train_data
      @data_struct
    end
  end
end
