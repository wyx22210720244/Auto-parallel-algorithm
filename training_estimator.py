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
    PARAM_BYTES = 2
    return args


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


def bytes_to_gb(bytes):
    return bytes / 1024 / 1024 / 1024


def get_transformer_block_param(args):
    h = args.hidden_size
    tp = args.tensor_model_parallel_size
    input_norm_w = h
    input_norm_b = h
    atten_qkv_w = 3 * h / tp * h
    atten_qkv_b = 3 * h / tp
    atten_dense_w = h * h / tp
    atten_dense_b = h
    post_atten_w = h
    post_atten_b = h
    mlp_h_4h_w = 4 * h / tp * h
    mlp_h_4h_b = 4 * h / tp
    mlp_4h_h_w = h * 4 * h / tp
    mlp_4h_h_b = h
    param = (input_norm_w + input_norm_b + atten_qkv_b + atten_qkv_w +
             atten_dense_w + atten_dense_b + post_atten_w + post_atten_b
             + mlp_h_4h_w + mlp_h_4h_b + mlp_4h_h_w + mlp_4h_h_b)
    return param


def get_embedding_param(args):
    h = args.hidden_size
    tp = args.tensor_model_parallel_size
    emb_w = 50304 / tp * h
    pos_w = args.seq_length * h
    return emb_w + pos_w


def get_final_norm_param(args):
    h = args.hidden_size
    tp = args.tensor_model_parallel_size
    final_norm_w = h
    final_norm_b = h
    return final_norm_b + final_norm_w


def get_final_embedding_param(args):
    h = args.hidden_size
    tp = args.tensor_model_parallel_size
    return 50304 / tp * h


def get_memory(args):
    global PARAM_BYTES
    t = args.tensor_model_parallel_size
    h = args.hidden_size
    s = args.seq_length
    b = args.micro_batch_size
    a = args.num_attention_heads
    n = args.num_layers
    param = (get_embedding_param(args) + get_transformer_block_param(args) * n + get_final_norm_param(
        args) + get_final_embedding_param(args)) * t
    act_mem_per_layer = s * b * h * (10 + 24 / t + 5 * a * s / (h * t))
    fixed_memory = bytes_to_gb(param * PARAM_BYTES * 8)
    variable_memory = bytes_to_gb(act_mem_per_layer * PARAM_BYTES * n)
    return (fixed_memory + variable_memory) * 1.2


def get_stage_memory(stage, args, max_stage, layers):
    global PARAM_BYTES
    t = args.tensor_model_parallel_size
    h = args.hidden_size
    s = args.seq_length
    b = args.micro_batch_size
    a = args.num_attention_heads
    l = layers
    param = get_transformer_block_param(args) * l * t
    # 判断首尾stage来添加格外参数
    if stage == 0:
        param += get_embedding_param(args)*t
    act_mem_per_layer = s * b * h * (10 + 24 / t + 5 * a * s / (h * t))
    fixed_memory = bytes_to_gb(param * PARAM_BYTES * 8)
    variable_memory = bytes_to_gb(act_mem_per_layer * l * (max_stage - stage) * PARAM_BYTES)
    return fixed_memory + variable_memory


def get_param(args):
    t = args.tensor_model_parallel_size
    n = args.num_layers
    param = (get_embedding_param(args) + get_transformer_block_param(args) * n + get_final_norm_param(
        args) + get_final_embedding_param(args)) * t
    return param


if __name__ == "__main__":
    args = get_args()
    print(f"估计显存值:{get_memory(args)}")
    print(f"参数量:{get_param(args)}")
