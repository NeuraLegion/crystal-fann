@[Link("fann")]
lib LibFANN
  ERRSTR_MAX = 128

  struct Error
    errno_f : ErrnoEnum
    error_log : File*
    errstr : LibC::Char*
  end

  enum ErrnoEnum
    ENoError                  =  0
    ECantOpenConfigR          =  1
    ECantOpenConfigW          =  2
    EWrongConfigVersion       =  3
    ECantReadConfig           =  4
    ECantReadNeuron           =  5
    ECantReadConnections      =  6
    EWrongNumConnections      =  7
    ECantOpenTdW              =  8
    ECantOpenTdR              =  9
    ECantReadTd               = 10
    ECantAllocateMem          = 11
    ECantTrainActivation      = 12
    ECantUseActivation        = 13
    ETrainDataMismatch        = 14
    ECantUseTrainAlg          = 15
    ETrainDataSubset          = 16
    EIndexOutOfBound          = 17
    EScaleNotPresent          = 18
    EInputNoMatch             = 19
    EOutputNoMatch            = 20
    EWrongParametersForCreate = 21
  end

  struct X_IoFile
    _flags : LibC::Int
    _io_read_ptr : LibC::Char*
    _io_read_end : LibC::Char*
    _io_read_base : LibC::Char*
    _io_write_base : LibC::Char*
    _io_write_ptr : LibC::Char*
    _io_write_end : LibC::Char*
    _io_buf_base : LibC::Char*
    _io_buf_end : LibC::Char*
    _io_save_base : LibC::Char*
    _io_backup_base : LibC::Char*
    _io_save_end : LibC::Char*
    _markers : X_IoMarker*
    _chain : X_IoFile*
    _fileno : LibC::Int
    _flags2 : LibC::Int
    _old_offset : X__OffT
    _cur_column : LibC::UShort
    _vtable_offset : LibC::Char
    _shortbuf : LibC::Char[1]
    _lock : X_IoLockT*
    _offset : X__Off64T
    __pad1 : Void*
    __pad2 : Void*
    __pad3 : Void*
    __pad4 : Void*
    __pad5 : LibC::Int
    _mode : LibC::Int
    _unused2 : LibC::Char
  end

  type File = X_IoFile

  struct X_IoMarker
    _next : X_IoMarker*
    _sbuf : X_IoFile*
    _pos : LibC::Int
  end

  alias X__OffT = LibC::Long
  alias X_IoLockT = Void
  alias X__Off64T = LibC::Long
  fun set_error_log = fann_set_error_log(errdat : Error*, log_file : File*)
  fun get_errno = fann_get_errno(errdat : Error*) : ErrnoEnum
  fun reset_errno = fann_reset_errno(errdat : Error*)
  fun reset_errstr = fann_reset_errstr(errdat : Error*)
  fun get_errstr = fann_get_errstr(errdat : Error*) : LibC::Char*
  fun print_error = fann_print_error(errdat : Error*)

  struct TrainData
    errno_f : ErrnoEnum
    error_log : File*
    errstr : LibC::Char*
    num_data : LibC::UInt
    num_input : LibC::UInt
    num_output : LibC::UInt
    input : Type**
    output : Type**
  end

  alias Type = LibC::Float

  struct Neuron
    first_con : LibC::UInt
    last_con : LibC::UInt
    sum : Type
    value : Type
    activation_steepness : Type
    activation_function : ActivationfuncEnum
  end

  enum ActivationfuncEnum
    Linear                   =  0
    Threshold                =  1
    ThresholdSymmetric       =  2
    Sigmoid                  =  3
    SigmoidStepwise          =  4
    SigmoidSymmetric         =  5
    SigmoidSymmetricStepwise =  6
    Gaussian                 =  7
    GaussianSymmetric        =  8
    GaussianStepwise         =  9
    Elliot                   = 10
    ElliotSymmetric          = 11
    LinearPiece              = 12
    LinearPieceSymmetric     = 13
    SinSymmetric             = 14
    CosSymmetric             = 15
    Sin                      = 16
    Cos                      = 17
    SigmoidSymmetricLecun    = 18
    Relu                     = 19
    LeakyRelu                = 20
  end

  struct Layer
    first_neuron : Neuron*
    last_neuron : Neuron*
  end

  struct Connection
    from_neuron : LibC::UInt
    to_neuron : LibC::UInt
    weight : Type
  end

  fun allocate_structure = fann_allocate_structure(num_layers : LibC::UInt) : Fann*

  struct Fann
    errno_f : ErrnoEnum
    error_log : File*
    errstr : LibC::Char*
    learning_rate : LibC::Double
    learning_momentum : LibC::Double
    learning_l2_norm : LibC::Double
    connection_rate : LibC::Double
    network_type : NettypeEnum
    first_layer : Layer*
    last_layer : Layer*
    total_neurons : LibC::UInt
    num_input : LibC::UInt
    num_output : LibC::UInt
    weights : Type*
    connections : Neuron**
    train_errors : Type*
    training_algorithm : TrainEnum
    total_connections : LibC::UInt
    output : Type*
    num_mse : LibC::UInt
    mse_value : LibC::Double
    num_bit_fail : LibC::UInt
    bit_fail_limit : Type
    train_error_function : ErrorfuncEnum
    train_stop_function : StopfuncEnum
    callback : CallbackType
    user_data : Void*
    cascade_output_change_fraction : LibC::Double
    cascade_output_stagnation_epochs : LibC::UInt
    cascade_candidate_change_fraction : LibC::Double
    cascade_candidate_stagnation_epochs : LibC::UInt
    cascade_best_candidate : LibC::UInt
    cascade_candidate_limit : Type
    cascade_weight_multiplier : Type
    cascade_max_out_epochs : LibC::UInt
    cascade_max_cand_epochs : LibC::UInt
    cascade_min_out_epochs : LibC::UInt
    cascade_min_cand_epochs : LibC::UInt
    cascade_activation_functions : ActivationfuncEnum*
    cascade_activation_functions_count : LibC::UInt
    cascade_activation_steepnesses : Type*
    cascade_activation_steepnesses_count : LibC::UInt
    cascade_num_candidate_groups : LibC::UInt
    cascade_candidate_scores : Type*
    total_neurons_allocated : LibC::UInt
    total_connections_allocated : LibC::UInt
    quickprop_decay : LibC::Double
    quickprop_mu : LibC::Double
    rprop_increase_factor : LibC::Double
    rprop_decrease_factor : LibC::Double
    rprop_delta_min : LibC::Double
    rprop_delta_max : LibC::Double
    rprop_delta_zero : LibC::Double
    sarprop_weight_decay_shift : LibC::Double
    sarprop_step_error_threshold_factor : LibC::Double
    sarprop_step_error_shift : LibC::Double
    sarprop_temperature : LibC::Double
    sarprop_epoch : LibC::UInt
    train_slopes : Type*
    prev_steps : Type*
    prev_train_slopes : Type*
    prev_weights_deltas : Type*
    scale_mean_in : LibC::Double*
    scale_deviation_in : LibC::Double*
    scale_new_min_in : LibC::Double*
    scale_factor_in : LibC::Double*
    scale_mean_out : LibC::Double*
    scale_deviation_out : LibC::Double*
    scale_new_min_out : LibC::Double*
    scale_factor_out : LibC::Double*
  end

  enum NettypeEnum
    NettypeLayer    = 0
    NettypeShortcut = 1
  end
  enum TrainEnum
    TrainIncremental = 0
    TrainBatch       = 1
    TrainRprop       = 2
    TrainQuickprop   = 3
    TrainSarprop     = 4
  end
  enum ErrorfuncEnum
    ErrorfuncLinear = 0
    ErrorfuncTanh   = 1
  end
  enum StopfuncEnum
    StopfuncMse = 0
    StopfuncBit = 1
  end
  alias CallbackType = (Fann*, TrainData*, LibC::UInt, LibC::UInt, LibC::Double, LibC::UInt -> LibC::Int)
  fun allocate_neurons = fann_allocate_neurons(ann : Fann*)
  fun allocate_connections = fann_allocate_connections(ann : Fann*)
  fun save_internal = fann_save_internal(ann : Fann*, configuration_file : LibC::Char*, save_as_fixed : LibC::UInt) : LibC::Int
  fun save_internal_fd = fann_save_internal_fd(ann : Fann*, conf : File*, configuration_file : LibC::Char*, save_as_fixed : LibC::UInt) : LibC::Int
  fun save_train_internal = fann_save_train_internal(data : TrainData*, filename : LibC::Char*, save_as_fixed : LibC::UInt, decimal_point : LibC::UInt) : LibC::Int
  fun save_train_internal_fd = fann_save_train_internal_fd(data : TrainData*, file : File*, filename : LibC::Char*, save_as_fixed : LibC::UInt, decimal_point : LibC::UInt) : LibC::Int
  fun update_stepwise = fann_update_stepwise(ann : Fann*)
  fun seed_rand = fann_seed_rand
  fun error = fann_error(errdat : Error*, errno_f : ErrnoEnum, ...)
  fun init_error_data = fann_init_error_data(errdat : Error*)
  fun create_from_fd = fann_create_from_fd(conf : File*, configuration_file : LibC::Char*) : Fann*
  fun read_train_from_fd = fann_read_train_from_fd(file : File*, filename : LibC::Char*) : TrainData*
  fun compute_mse = fann_compute_MSE(ann : Fann*, desired_output : Type*)
  fun update_output_weights = fann_update_output_weights(ann : Fann*)
  fun backpropagate_mse = fann_backpropagate_MSE(ann : Fann*)
  fun update_weights = fann_update_weights(ann : Fann*)
  fun update_slopes_batch = fann_update_slopes_batch(ann : Fann*, layer_begin : Layer*, layer_end : Layer*)
  fun update_weights_quickprop = fann_update_weights_quickprop(ann : Fann*, num_data : LibC::UInt, first_weight : LibC::UInt, past_end : LibC::UInt)
  fun update_weights_batch = fann_update_weights_batch(ann : Fann*, num_data : LibC::UInt, first_weight : LibC::UInt, past_end : LibC::UInt)
  fun update_weights_irpropm = fann_update_weights_irpropm(ann : Fann*, first_weight : LibC::UInt, past_end : LibC::UInt)
  fun update_weights_sarprop = fann_update_weights_sarprop(ann : Fann*, epoch : LibC::UInt, first_weight : LibC::UInt, past_end : LibC::UInt)
  fun clear_train_arrays = fann_clear_train_arrays(ann : Fann*)
  fun activation = fann_activation(ann : Fann*, activation_function : LibC::UInt, steepness : Type, value : Type) : Type
  fun activation_derived = fann_activation_derived(activation_function : LibC::UInt, steepness : Type, value : Type, sum : Type) : Type
  fun desired_error_reached = fann_desired_error_reached(ann : Fann*, desired_error : LibC::Double) : LibC::Int
  fun train_outputs = fann_train_outputs(ann : Fann*, data : TrainData*, desired_error : LibC::Double) : LibC::Int
  fun train_outputs_epoch = fann_train_outputs_epoch(ann : Fann*, data : TrainData*) : LibC::Double
  fun train_candidates = fann_train_candidates(ann : Fann*, data : TrainData*) : LibC::Int
  fun train_candidates_epoch = fann_train_candidates_epoch(ann : Fann*, data : TrainData*) : Type
  fun install_candidate = fann_install_candidate(ann : Fann*)
  fun check_input_output_sizes = fann_check_input_output_sizes(ann : Fann*, data : TrainData*) : LibC::Int
  fun initialize_candidates = fann_initialize_candidates(ann : Fann*) : LibC::Int
  fun set_shortcut_connections = fann_set_shortcut_connections(ann : Fann*)
  fun allocate_scale = fann_allocate_scale(ann : Fann*) : LibC::Int
  fun scale_data_to_range = fann_scale_data_to_range(data : Type**, num_data : LibC::UInt, num_elem : LibC::UInt, old_min : Type, old_max : Type, new_min : Type, new_max : Type)
  fun train = fann_train(ann : Fann*, input : Type*, desired_output : Type*)
  fun test = fann_test(ann : Fann*, input : Type*, desired_output : Type*) : Type*
  fun get_mse = fann_get_MSE(ann : Fann*) : LibC::Double
  fun get_bit_fail = fann_get_bit_fail(ann : Fann*) : LibC::UInt
  fun reset_mse = fann_reset_MSE(ann : Fann*)
  fun train_on_data = fann_train_on_data(ann : Fann*, data : TrainData*, max_epochs : LibC::UInt, epochs_between_reports : LibC::UInt, desired_error : LibC::Double)
  fun train_on_file = fann_train_on_file(ann : Fann*, filename : LibC::Char*, max_epochs : LibC::UInt, epochs_between_reports : LibC::UInt, desired_error : LibC::Double)
  fun train_epoch = fann_train_epoch(ann : Fann*, data : TrainData*) : LibC::Double
  fun train_epoch_lw = fann_train_epoch_lw(ann : Fann*, data : TrainData*, label_weight : Type*) : LibC::Double
  fun train_epoch_irpropm_gradient = fann_train_epoch_irpropm_gradient(ann : Fann*, data : TrainData*, error_function : (Type*, Type*, LibC::Int, Void* -> Type), x3 : Void*) : LibC::Double
  fun compute_mse_gradient = fann_compute_MSE_gradient(x0 : Fann*, x1 : Type*, error_function : (Type*, Type*, LibC::Int, Void* -> Type), x3 : Void*)
  fun backpropagate_mse_firstlayer = fann_backpropagate_MSE_firstlayer(x0 : Fann*)
  fun test_data = fann_test_data(ann : Fann*, data : TrainData*) : LibC::Double
  fun read_train_from_file = fann_read_train_from_file(filename : LibC::Char*) : TrainData*
  fun create_train = fann_create_train(num_data : LibC::UInt, num_input : LibC::UInt, num_output : LibC::UInt) : TrainData*
  fun create_train_pointer_array = fann_create_train_pointer_array(num_data : LibC::UInt, num_input : LibC::UInt, input : Type**, num_output : LibC::UInt, output : Type**) : TrainData*
  fun create_train_array = fann_create_train_array(num_data : LibC::UInt, num_input : LibC::UInt, input : Type*, num_output : LibC::UInt, output : Type*) : TrainData*
  fun create_train_from_callback = fann_create_train_from_callback(num_data : LibC::UInt, num_input : LibC::UInt, num_output : LibC::UInt, user_function : (LibC::UInt, LibC::UInt, LibC::UInt, Type*, Type* -> Void)) : TrainData*
  fun destroy_train = fann_destroy_train(train_data : TrainData*)
  fun get_train_input = fann_get_train_input(data : TrainData*, position : LibC::UInt) : Type*
  fun get_train_output = fann_get_train_output(data : TrainData*, position : LibC::UInt) : Type*
  fun shuffle_train_data = fann_shuffle_train_data(train_data : TrainData*)
  fun get_min_train_input = fann_get_min_train_input(train_data : TrainData*) : Type
  fun get_max_train_input = fann_get_max_train_input(train_data : TrainData*) : Type
  fun get_min_train_output = fann_get_min_train_output(train_data : TrainData*) : Type
  fun get_max_train_output = fann_get_max_train_output(train_data : TrainData*) : Type
  fun scale_train = fann_scale_train(ann : Fann*, data : TrainData*)
  fun descale_train = fann_descale_train(ann : Fann*, data : TrainData*)
  fun set_input_scaling_params = fann_set_input_scaling_params(ann : Fann*, data : TrainData*, new_input_min : LibC::Double, new_input_max : LibC::Double) : LibC::Int
  fun set_output_scaling_params = fann_set_output_scaling_params(ann : Fann*, data : TrainData*, new_output_min : LibC::Double, new_output_max : LibC::Double) : LibC::Int
  fun set_scaling_params = fann_set_scaling_params(ann : Fann*, data : TrainData*, new_input_min : LibC::Double, new_input_max : LibC::Double, new_output_min : LibC::Double, new_output_max : LibC::Double) : LibC::Int
  fun clear_scaling_params = fann_clear_scaling_params(ann : Fann*) : LibC::Int
  fun scale_input = fann_scale_input(ann : Fann*, input_vector : Type*)
  fun scale_output = fann_scale_output(ann : Fann*, output_vector : Type*)
  fun descale_input = fann_descale_input(ann : Fann*, input_vector : Type*)
  fun descale_output = fann_descale_output(ann : Fann*, output_vector : Type*)
  fun scale_input_train_data = fann_scale_input_train_data(train_data : TrainData*, new_min : Type, new_max : Type)
  fun scale_output_train_data = fann_scale_output_train_data(train_data : TrainData*, new_min : Type, new_max : Type)
  fun scale_train_data = fann_scale_train_data(train_data : TrainData*, new_min : Type, new_max : Type)
  fun merge_train_data = fann_merge_train_data(data1 : TrainData*, data2 : TrainData*) : TrainData*
  fun duplicate_train_data = fann_duplicate_train_data(data : TrainData*) : TrainData*
  fun subset_train_data = fann_subset_train_data(data : TrainData*, pos : LibC::UInt, length : LibC::UInt) : TrainData*
  fun length_train_data = fann_length_train_data(data : TrainData*) : LibC::UInt
  fun num_input_train_data = fann_num_input_train_data(data : TrainData*) : LibC::UInt
  fun num_output_train_data = fann_num_output_train_data(data : TrainData*) : LibC::UInt
  fun save_train = fann_save_train(data : TrainData*, filename : LibC::Char*) : LibC::Int
  fun save_train_to_fixed = fann_save_train_to_fixed(data : TrainData*, filename : LibC::Char*, decimal_point : LibC::UInt) : LibC::Int
  fun get_training_algorithm = fann_get_training_algorithm(ann : Fann*) : TrainEnum
  fun set_training_algorithm = fann_set_training_algorithm(ann : Fann*, training_algorithm : TrainEnum)
  fun get_learning_rate = fann_get_learning_rate(ann : Fann*) : LibC::Double
  fun set_learning_rate = fann_set_learning_rate(ann : Fann*, learning_rate : LibC::Double)
  fun get_learning_momentum = fann_get_learning_momentum(ann : Fann*) : LibC::Double
  fun set_learning_momentum = fann_set_learning_momentum(ann : Fann*, learning_momentum : LibC::Double)
  fun get_learning_l2_norm = fann_get_learning_l2_norm(ann : Fann*) : LibC::Double
  fun set_learning_l2_norm = fann_set_learning_l2_norm(ann : Fann*, learning_l2_norm : LibC::Double)
  fun get_activation_function = fann_get_activation_function(ann : Fann*, layer : LibC::Int, neuron : LibC::Int) : ActivationfuncEnum
  fun set_activation_function = fann_set_activation_function(ann : Fann*, activation_function : ActivationfuncEnum, layer : LibC::Int, neuron : LibC::Int)
  fun set_activation_function_layer = fann_set_activation_function_layer(ann : Fann*, activation_function : ActivationfuncEnum, layer : LibC::Int)
  fun set_activation_function_hidden = fann_set_activation_function_hidden(ann : Fann*, activation_function : ActivationfuncEnum)
  fun set_activation_function_output = fann_set_activation_function_output(ann : Fann*, activation_function : ActivationfuncEnum)
  fun get_activation_steepness = fann_get_activation_steepness(ann : Fann*, layer : LibC::Int, neuron : LibC::Int) : Type
  fun set_activation_steepness = fann_set_activation_steepness(ann : Fann*, steepness : Type, layer : LibC::Int, neuron : LibC::Int)
  fun set_activation_steepness_layer = fann_set_activation_steepness_layer(ann : Fann*, steepness : Type, layer : LibC::Int)
  fun set_activation_steepness_hidden = fann_set_activation_steepness_hidden(ann : Fann*, steepness : Type)
  fun set_activation_steepness_output = fann_set_activation_steepness_output(ann : Fann*, steepness : Type)
  fun get_train_error_function = fann_get_train_error_function(ann : Fann*) : ErrorfuncEnum
  fun set_train_error_function = fann_set_train_error_function(ann : Fann*, train_error_function : ErrorfuncEnum)
  fun get_train_stop_function = fann_get_train_stop_function(ann : Fann*) : StopfuncEnum
  fun set_train_stop_function = fann_set_train_stop_function(ann : Fann*, train_stop_function : StopfuncEnum)
  fun get_bit_fail_limit = fann_get_bit_fail_limit(ann : Fann*) : Type
  fun set_bit_fail_limit = fann_set_bit_fail_limit(ann : Fann*, bit_fail_limit : Type)
  fun set_callback = fann_set_callback(ann : Fann*, callback : CallbackType)
  fun get_quickprop_decay = fann_get_quickprop_decay(ann : Fann*) : LibC::Double
  fun set_quickprop_decay = fann_set_quickprop_decay(ann : Fann*, quickprop_decay : LibC::Double)
  fun get_quickprop_mu = fann_get_quickprop_mu(ann : Fann*) : LibC::Double
  fun set_quickprop_mu = fann_set_quickprop_mu(ann : Fann*, quickprop_mu : LibC::Double)
  fun get_rprop_increase_factor = fann_get_rprop_increase_factor(ann : Fann*) : LibC::Double
  fun set_rprop_increase_factor = fann_set_rprop_increase_factor(ann : Fann*, rprop_increase_factor : LibC::Double)
  fun get_rprop_decrease_factor = fann_get_rprop_decrease_factor(ann : Fann*) : LibC::Double
  fun set_rprop_decrease_factor = fann_set_rprop_decrease_factor(ann : Fann*, rprop_decrease_factor : LibC::Double)
  fun get_rprop_delta_min = fann_get_rprop_delta_min(ann : Fann*) : LibC::Double
  fun set_rprop_delta_min = fann_set_rprop_delta_min(ann : Fann*, rprop_delta_min : LibC::Double)
  fun get_rprop_delta_max = fann_get_rprop_delta_max(ann : Fann*) : LibC::Double
  fun set_rprop_delta_max = fann_set_rprop_delta_max(ann : Fann*, rprop_delta_max : LibC::Double)
  fun get_rprop_delta_zero = fann_get_rprop_delta_zero(ann : Fann*) : LibC::Double
  fun set_rprop_delta_zero = fann_set_rprop_delta_zero(ann : Fann*, rprop_delta_max : LibC::Double)
  fun get_sarprop_weight_decay_shift = fann_get_sarprop_weight_decay_shift(ann : Fann*) : LibC::Double
  fun set_sarprop_weight_decay_shift = fann_set_sarprop_weight_decay_shift(ann : Fann*, sarprop_weight_decay_shift : LibC::Double)
  fun get_sarprop_step_error_threshold_factor = fann_get_sarprop_step_error_threshold_factor(ann : Fann*) : LibC::Double
  fun set_sarprop_step_error_threshold_factor = fann_set_sarprop_step_error_threshold_factor(ann : Fann*, sarprop_step_error_threshold_factor : LibC::Double)
  fun get_sarprop_step_error_shift = fann_get_sarprop_step_error_shift(ann : Fann*) : LibC::Double
  fun set_sarprop_step_error_shift = fann_set_sarprop_step_error_shift(ann : Fann*, sarprop_step_error_shift : LibC::Double)
  fun get_sarprop_temperature = fann_get_sarprop_temperature(ann : Fann*) : LibC::Double
  fun set_sarprop_temperature = fann_set_sarprop_temperature(ann : Fann*, sarprop_temperature : LibC::Double)
  fun cascadetrain_on_data = fann_cascadetrain_on_data(ann : Fann*, data : TrainData*, max_neurons : LibC::UInt, neurons_between_reports : LibC::UInt, desired_error : LibC::Double)
  fun cascadetrain_on_file = fann_cascadetrain_on_file(ann : Fann*, filename : LibC::Char*, max_neurons : LibC::UInt, neurons_between_reports : LibC::UInt, desired_error : LibC::Double)
  fun get_cascade_output_change_fraction = fann_get_cascade_output_change_fraction(ann : Fann*) : LibC::Double
  fun set_cascade_output_change_fraction = fann_set_cascade_output_change_fraction(ann : Fann*, cascade_output_change_fraction : LibC::Double)
  fun get_cascade_output_stagnation_epochs = fann_get_cascade_output_stagnation_epochs(ann : Fann*) : LibC::UInt
  fun set_cascade_output_stagnation_epochs = fann_set_cascade_output_stagnation_epochs(ann : Fann*, cascade_output_stagnation_epochs : LibC::UInt)
  fun get_cascade_candidate_change_fraction = fann_get_cascade_candidate_change_fraction(ann : Fann*) : LibC::Double
  fun set_cascade_candidate_change_fraction = fann_set_cascade_candidate_change_fraction(ann : Fann*, cascade_candidate_change_fraction : LibC::Double)
  fun get_cascade_candidate_stagnation_epochs = fann_get_cascade_candidate_stagnation_epochs(ann : Fann*) : LibC::UInt
  fun set_cascade_candidate_stagnation_epochs = fann_set_cascade_candidate_stagnation_epochs(ann : Fann*, cascade_candidate_stagnation_epochs : LibC::UInt)
  fun get_cascade_weight_multiplier = fann_get_cascade_weight_multiplier(ann : Fann*) : Type
  fun set_cascade_weight_multiplier = fann_set_cascade_weight_multiplier(ann : Fann*, cascade_weight_multiplier : Type)
  fun get_cascade_candidate_limit = fann_get_cascade_candidate_limit(ann : Fann*) : Type
  fun set_cascade_candidate_limit = fann_set_cascade_candidate_limit(ann : Fann*, cascade_candidate_limit : Type)
  fun get_cascade_max_out_epochs = fann_get_cascade_max_out_epochs(ann : Fann*) : LibC::UInt
  fun set_cascade_max_out_epochs = fann_set_cascade_max_out_epochs(ann : Fann*, cascade_max_out_epochs : LibC::UInt)
  fun get_cascade_min_out_epochs = fann_get_cascade_min_out_epochs(ann : Fann*) : LibC::UInt
  fun set_cascade_min_out_epochs = fann_set_cascade_min_out_epochs(ann : Fann*, cascade_min_out_epochs : LibC::UInt)
  fun get_cascade_max_cand_epochs = fann_get_cascade_max_cand_epochs(ann : Fann*) : LibC::UInt
  fun set_cascade_max_cand_epochs = fann_set_cascade_max_cand_epochs(ann : Fann*, cascade_max_cand_epochs : LibC::UInt)
  fun get_cascade_min_cand_epochs = fann_get_cascade_min_cand_epochs(ann : Fann*) : LibC::UInt
  fun set_cascade_min_cand_epochs = fann_set_cascade_min_cand_epochs(ann : Fann*, cascade_min_cand_epochs : LibC::UInt)
  fun get_cascade_num_candidates = fann_get_cascade_num_candidates(ann : Fann*) : LibC::UInt
  fun get_cascade_activation_functions_count = fann_get_cascade_activation_functions_count(ann : Fann*) : LibC::UInt
  fun get_cascade_activation_functions = fann_get_cascade_activation_functions(ann : Fann*) : ActivationfuncEnum*
  fun set_cascade_activation_functions = fann_set_cascade_activation_functions(ann : Fann*, cascade_activation_functions : ActivationfuncEnum*, cascade_activation_functions_count : LibC::UInt)
  fun get_cascade_activation_steepnesses_count = fann_get_cascade_activation_steepnesses_count(ann : Fann*) : LibC::UInt
  fun get_cascade_activation_steepnesses = fann_get_cascade_activation_steepnesses(ann : Fann*) : Type*
  fun set_cascade_activation_steepnesses = fann_set_cascade_activation_steepnesses(ann : Fann*, cascade_activation_steepnesses : Type*, cascade_activation_steepnesses_count : LibC::UInt)
  fun get_cascade_num_candidate_groups = fann_get_cascade_num_candidate_groups(ann : Fann*) : LibC::UInt
  fun set_cascade_num_candidate_groups = fann_set_cascade_num_candidate_groups(ann : Fann*, cascade_num_candidate_groups : LibC::UInt)
  fun create_from_file = fann_create_from_file(configuration_file : LibC::Char*) : Fann*
  fun save = fann_save(ann : Fann*, configuration_file : LibC::Char*) : LibC::Int
  fun save_to_fixed = fann_save_to_fixed(ann : Fann*, configuration_file : LibC::Char*) : LibC::Int
  fun create_standard = fann_create_standard(num_layers : LibC::UInt, ...) : Fann*
  fun create_standard_array = fann_create_standard_array(num_layers : LibC::UInt, layers : LibC::UInt*) : Fann*
  fun create_sparse = fann_create_sparse(connection_rate : LibC::Double, num_layers : LibC::UInt, ...) : Fann*
  fun create_sparse_array = fann_create_sparse_array(connection_rate : LibC::Double, num_layers : LibC::UInt, layers : LibC::UInt*) : Fann*
  fun create_shortcut = fann_create_shortcut(num_layers : LibC::UInt, ...) : Fann*
  fun create_shortcut_array = fann_create_shortcut_array(num_layers : LibC::UInt, layers : LibC::UInt*) : Fann*
  fun destroy = fann_destroy(ann : Fann*)
  fun copy = fann_copy(ann : Fann*) : Fann*
  fun run = fann_run(ann : Fann*, input : Type*) : Type*
  fun randomize_weights = fann_randomize_weights(ann : Fann*, min_weight : Type, max_weight : Type)
  fun init_weights = fann_init_weights(ann : Fann*, train_data : TrainData*)
  fun print_connections = fann_print_connections(ann : Fann*)
  fun print_parameters = fann_print_parameters(ann : Fann*)
  fun get_num_input = fann_get_num_input(ann : Fann*) : LibC::UInt
  fun get_num_output = fann_get_num_output(ann : Fann*) : LibC::UInt
  fun get_total_neurons = fann_get_total_neurons(ann : Fann*) : LibC::UInt
  fun get_total_connections = fann_get_total_connections(ann : Fann*) : LibC::UInt
  fun get_network_type = fann_get_network_type(ann : Fann*) : NettypeEnum
  fun get_connection_rate = fann_get_connection_rate(ann : Fann*) : LibC::Double
  fun get_num_layers = fann_get_num_layers(ann : Fann*) : LibC::UInt
  fun get_layer_array = fann_get_layer_array(ann : Fann*, layers : LibC::UInt*)
  fun get_bias_array = fann_get_bias_array(ann : Fann*, bias : LibC::UInt*)
  fun get_connection_array = fann_get_connection_array(ann : Fann*, connections : Connection*)
  fun get_l1_norm = fann_get_l1_norm(ann : Fann*) : Type
  fun get_l2_norm = fann_get_l2_norm(ann : Fann*) : Type
  fun set_weight_array = fann_set_weight_array(ann : Fann*, connections : Connection*, num_connections : LibC::UInt)
  fun set_weight = fann_set_weight(ann : Fann*, from_neuron : LibC::UInt, to_neuron : LibC::UInt, weight : Type)
  fun get_weights = fann_get_weights(ann : Fann*, weights : Type*)
  fun set_weights = fann_set_weights(ann : Fann*, weights : Type*)
  fun set_user_data = fann_set_user_data(ann : Fann*, user_data : Void*)
  fun get_user_data = fann_get_user_data(ann : Fann*) : Void*
  fun disable_seed_rand = fann_disable_seed_rand
  fun enable_seed_rand = fann_enable_seed_rand
  $default_error_log : File*
  $train_names = TRAIN_NAMES : LibC::Char*[5]
  $activationfunc_names = ACTIVATIONFUNC_NAMES : LibC::Char*[21]
  $errorfunc_names = ERRORFUNC_NAMES : LibC::Char*[2]
  $stopfunc_names = STOPFUNC_NAMES : LibC::Char*[2]
  $nettype_names = NETTYPE_NAMES : LibC::Char*[2]
end
