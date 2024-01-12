import argparse
import json

global PARAM_BYTES


def get_args():
    parser = argparse.ArgumentParser(description='Training Estimator')
    parser.add_argument("--tensor-model-parallel-size",
                        type=int,
                        default=1)
    parser.add_argument("--pipeline-model-parallel-size",
                        type=int,
                        default=1)
    parser.add_argument("--data-parallel-size",
                        type=int, )
    parser.add_argument("--sequence_parallel",
                        type=bool,
                        default=False)
    parser.add_argument('--num-layers',
                        type=int,
                        default=None)
    parser.add_argument('--hidden-size',
                        type=int,
                        default=None)
    parser.add_argument('--ffn-hidden-size',
                        type=int,
                        default=None)
    parser.add_argument('--attention-size',
                        type=int,
                        default=None)
    parser.add_argument('--num-attention-heads',
                        type=int,
                        default=None)
    parser.add_argument('--max-position-embeddings',
                        type=int,
                        default=None)
    parser.add_argument('--seq-length',
                        type=int,
                        default=None)
    parser.add_argument('--micro-batch-size',
                        type=int,
                        default=None)
    parser.add_argument('--global-batch-size',
                        type=int,
                        default=None)
    parser.add_argument('--vocab-file',
                        type=str,
                        default=None)
    parser.add_argument('--nproc_per_node',
                        type=int,
                        default=None)
    parser.add_argument('--nnodes',
                        type=int,
                        default=None)
    parser.add_argument('--world_size',
                        type=int)
    parser.add_argument('--fp16',
                        action='store_true', )
    parser.add_argument('use_flash_attn',
                        action='store_true')
    parser.add_argument('--use-distributed-optimizer',
                        action='store_true')
    parser.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                        help='Pad the vocab size to be divisible by this value.'
                             'This is added for computational efficieny reasons.')
    parser.add_argument('--swiglu',
                        action='store_true')
    args, undefined = parser.parse_known_args()
    if undefined is not None:
        print(f"undefined variable：{undefined}")
        print("Please check the input parameters")
        print("==================================")
    # args.world_size = args.nproc_per_node * args.nnodes
    # args.data_parallel_size = args.world_size // (args.tensor_model_parallel_size * args.pipeline_model_parallel_size)
    global PARAM_BYTES
    PARAM_BYTES = 2 if args.fp16 or args.bf16 else 4
    return args


def get_ffn_hidden_size(args):
    if args.ffn_hidden_size is None:
        if args.swiglu:
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
        else:
            args.ffn_hidden_size = 4 * args.hidden_size
    return args.ffn_hidden_size


def get_vocab_size_with_padding(args):
    if args.vocab_file is not None:
        vocab_size = len(json.load(open(args.vocab_file)))
        multiple = args.make_vocab_size_divisible_by * \
                   args.tensor_model_parallel_size
        while (vocab_size % multiple) != 0:
            vocab_size += 1
    else:
        raise ValueError("vocab_file is None")
    return vocab_size


def get_embedding_parameter(args):
    vocab_size = get_vocab_size_with_padding(args)
    # vocab_size = 130528
    embedding_parameter = (vocab_size + args.seq_length) * args.hidden_size
    return embedding_parameter


def get_transformer_parameter(args):
    attention_parameter = (args.hidden_size * args.hidden_size +
                           args.hidden_size) * 4 + args.hidden_size * 2
    return attention_parameter


def get_attention_size(args):
    args.attention_size = args.hidden_size // args.num_attention_heads
    return args.attention_size


def get_ffn_parameter(args):
    ffn_hidden_size = get_ffn_hidden_size(args)
    ffn_parameter = (ffn_hidden_size * args.hidden_size * 2
                     + args.hidden_size * 3 + ffn_hidden_size)
    return ffn_parameter


def get_full_model_parameter(args):
    embedding_parameter = get_embedding_parameter(args)
    transformer_parameter = get_transformer_parameter(args)
    ffn_parameter = get_ffn_parameter(args)
    full_model_parameter = embedding_parameter + \
                           (transformer_parameter + ffn_parameter) * args.num_layers
    return full_model_parameter


def get_activation_memory(args):
    batch_size = args.micro_batch_size
    num_layer = args.num_layers
    seq_length = args.seq_length
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    ffn_size = get_ffn_hidden_size(args)
    global PARAM_BYTES
    activation_memory = num_layer * batch_size * (PARAM_BYTES * (10 * seq_length * hidden_size
                                                                 + 2 * seq_length * ffn_size + 2 * seq_length * seq_length * num_attention_heads)
                                                  + seq_length * seq_length * args.num_attention_heads + 2 * seq_length * hidden_size)

    return activation_memory


def bytes_to_gb(bytes):
    return bytes / 1024 / 1024 / 1024


def get_parameter_memory(args):
    global PARAM_BYTES
    parameter_memory = get_full_model_parameter(args) * PARAM_BYTES
    return parameter_memory


def get_gradient_memory(args):
    global PARAM_BYTES
    gradient_memory = get_full_model_parameter(args) * PARAM_BYTES
    if args.pipeline_model_parallel_size > 1:
        gradient_memory = gradient_memory * 2
    return gradient_memory


def get_optimizer_memory(args):
    global PARAM_BYTES
    # 以Adam进行计算
    if PARAM_BYTES == 2:
        optimizer_memory = get_full_model_parameter(args) * PARAM_BYTES * 6
    else:
        optimizer_memory = get_full_model_parameter(args) * PARAM_BYTES * 2
    return optimizer_memory


def get_peak_memory(args):
    peak_memory = get_parameter_memory(args) + \
                  get_optimizer_memory(args) + \
                  get_gradient_memory(args) + \
                  get_activation_memory(args)
    return peak_memory


def get_memory_without_activation(args):
    memory_without_activation = get_parameter_memory(args) + \
                                get_optimizer_memory(args) + \
                                get_gradient_memory(args)
    return memory_without_activation


def get_activation_memory_with_pp_and_tp(args):
    pp_size = args.pipeline_model_parallel_size
    # tp_size = args.tensor_model_parallel_size
    activation_per_layer = get_activation_memory(args) / args.num_layers
    total_activation_memory = 0
    for i in range(pp_size):
        total_activation_memory += activation_per_layer * args.num_layers / pp_size * (pp_size - i)
        # print(f"total_activation_memory_tem:{total_activation_memory}")
    # total_activation_memory = activation_per_layer * args.num_layers * (pp_size - 1 + 1 / pp_size)
    return total_activation_memory


def get_activation_memory_for_first_pp_stage(args):
    activation_per_layer = get_activation_memory(args) / args.num_layers
    activation_memory_for_first_stage = activation_per_layer / args.tensor_model_parallel_size * args.num_layers
    return activation_memory_for_first_stage


def get_full_training_memory_consumption(args):
    memory_consumption = get_activation_memory_with_pp_and_tp(args) + get_memory_without_activation(args)
    return bytes_to_gb(memory_consumption)


if __name__ == "__main__":
    args = get_args()
    embedding_parameter = get_embedding_parameter(args)
    transformer_parameter = get_transformer_parameter(args)
    ffn_parameter = get_ffn_parameter(args)
    full_model_parameter = get_full_model_parameter(args)
    print(f"embedding_parameter:{embedding_parameter}")
    print(f"transformer_parameter:{transformer_parameter}")
    print(f"ffn_parameter:{ffn_parameter}")
    print(f"full_model_parameter:{full_model_parameter}")
    print(f"full_activation_memory:{bytes_to_gb(get_activation_memory(args))}GB")
    # print(f"peak_memory:{bytes_to_gb(get_peak_memory(args))}GB")
    print(f"memory_for_first_pp_stage:{bytes_to_gb(get_activation_memory_for_first_pp_stage(args))}GB")
    print(f"memory_without_activation:{bytes_to_gb(get_memory_without_activation(args))}GB")
    print(f"activation_memory_with_pp_and_tp:{bytes_to_gb(get_activation_memory_with_pp_and_tp(args))}GB")
    print(
        f"total_memory:{bytes_to_gb(get_activation_memory_with_pp_and_tp(args) + get_memory_without_activation(args))}GB")
