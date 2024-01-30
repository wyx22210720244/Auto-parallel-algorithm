import torch
import torch.distributed as dist
import json
from gpu import GPU, Cluster, NODE
from collections import defaultdict

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# tensor model parallel group and data parallel group combined
# used for fp8 and moe training
_TENSOR_AND_DATA_PARALLEL_GROUP = None
# Expert parallel group that the current rank belongs to.
_TENSOR_AND_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None
# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel_ranks
_CONTEXT_PARALLEL_GLOBAL_RANKS = None

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

# combined parallel group of TP, DP, and CP used for fp8
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None


def test_all_reduce():
    dist.init_process_group(backend='gloo', init_method='env://')
    rank = dist.get_rank()
    tensor = torch.tensor([rank], dtype=torch.float32)
    print(f'Rank {rank} has original tensor: {tensor}')
    if rank < 2:
        group = dist.new_group([0, 1])
    else:
        group = dist.new_group([2, 3])
    dist.barrier()
    dist.all_reduce(tensor, group=group)
    print(f'Rank {rank} has tensor after all reduce: {tensor}')


def dict_to_json(dict):
    result = defaultdict(list)
    for key, value in dict.items():
        for gpu in value:
            if isinstance(gpu, list):
                result[key].append([(gpu[0].node, gpu[0].local_rank), (gpu[1].node, gpu[1].local_rank)])
            else:
                result[key].append((gpu.node, gpu.local_rank))
    return result


def test_group_partition():
    data_parallel_size = 0
    with open('allocations.json', 'r') as f:
        group_allocation = json.load(f)
    for _, _ in group_allocation.items():
        data_parallel_size += 1
    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for _, value in group_allocation.items():
        model_ranks = []
        for gpu in value:
            if isinstance(gpu, list):
                model_ranks.extend([gpu[0], gpu[1]])
            else:
                model_ranks.append(gpu)
        print(f'model_ranks: {sorted(model_ranks)}')

        # group = torch.distributed.new_group(
        #     model_ranks, pg_options=get_nccl_options('mp', nccl_comm_cfgs)
        # )
        # if rank in model_ranks:
        #     _MODEL_PARALLEL_GROUP = group
        # print(f'key: {key}, model_rank: {model_ranks}')

        # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert (
            _TENSOR_MODEL_PARALLEL_GROUP is None
    ), 'tensor model parallel group is already initialized'
    for key, value in group_allocation.items():
        tensor_ranks = []
        for gpu in value:
            if isinstance(gpu, list):
                group = torch.distributed.new_group(
                    gpu, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
                )
        if rank in gpu:
            _TENSOR_MODEL_PARALLEL_GROUP = group
        # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
            _PIPELINE_MODEL_PARALLEL_GROUP is None
    ), 'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'
    for key, value in group_allocation.items():
        pipeline_ranks = []
        for gpu in value:
            if isinstance(gpu, list):
                pipeline_ranks.append(gpu[0])
            else:
                pipeline_ranks.append(gpu)
            # if rank in pipeline_ranks:
            #     _PIPELINE_MODEL_PARALLEL_GROUP = group
            #     _PIPELINE_GLOBAL_RANKS = pipeline_ranks
            #     _EMBEDDING_GROUP = group
            #     _EMBEDDING_GLOBAL_RANKS = pipeline_ranks[:1]
            #     _POSITION_EMBEDDING_GROUP = group
            #     _POSITION_EMBEDDING_GLOBAL_RANKS = pipeline_ranks[-1:]
            # print(f'key: {key}, pipeline_rank: {pipeline_ranks}')


def test_json_loading():
    with open('allocations.json', 'r') as f:
        group_allocation = json.load(f)
    for key, value in group_allocation.items():
        print(f'key: {key}, value: {value}')


def test_origin_group_allocation():
    world_size = 16
    tensor_model_parallel_size = 2  # 2 GPUs to parallelize the model tensor
    pipeline_model_parallel_size = 4  # 4 GPUs to parallelize the model pipeline
    data_parallel_size = world_size // (tensor_model_parallel_size *
                                        pipeline_model_parallel_size)  # 1
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size  # 4
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size  # 2
    num_data_parallel_groups = world_size // data_parallel_size  # 8

    # Build the data-parallel groups.
    print("------ Build the data-parallel groups -----")
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank,
                          tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
    print(all_data_parallel_group_ranks)

    # Build the model-parallel groups.
    print("------ Build the model-parallel groups -----")
    for i in range(data_parallel_size):
        ranks = [data_parallel_group_ranks[i]
                 for data_parallel_group_ranks in all_data_parallel_group_ranks]
        print(f"pp_group{list(ranks)}")
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks
        print(f'embedding_ranks: {embedding_ranks}')
        print(f'position_embedding_ranks: {position_embedding_ranks}')
    # Build the tensor model-parallel groups.
    print("------ Build the tensor model-parallel groups -----")
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size,
                      (i + 1) * tensor_model_parallel_size)
        print(list(ranks))

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    print("------ Build the pipeline model-parallel groups -----")
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size,
                      num_pipeline_model_parallel_groups)
        print(list(ranks))

    print("----------context parallel group----------")
    context_parallel_size = 1
    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                print(list(ranks))


def test_zip(list_var):
    for idx, element in enumerate(zip(*list_var)):
        print(idx)
        print(element)


def test_change_to_symmetric_list(list1):
    list_var = [[item] if not isinstance(item, list) else item for item in list1]
    for value in list_var:
        if len(value) == 1:
            value.append(value[0])
    return list_var

def test_torch_empty():
    tensor  = torch.randn(10,5)
    print(tensor)
    print(tensor[:5, :])

def test_dict_identity():
    dict = {'a': 1, 'b': 1}
    print(dict['a']==dict['b'])



def test_in_list():
    list1 = [1,2,3,4]
    print(4 in list1)


def test_silce():
    tensor = torch.randn(10, 5)
    tensor2 = tensor[:5, :]
    print(tensor)
    print(tensor2)
    tensor2[0, 0] = 0
    print(tensor)

def test_dp_layer_rank_mapping():
    map = {"1":{0:[1,2],1:[3,4]}, "2":{2:[1,4]}}
    filename = "dp_layer_rank_mapping.json"
    with open(filename, 'w') as f:
        json.dump(map, f)
    # result = json.dumps(map)

if __name__ == '__main__':
    # test_origin_group_allocation()
    # test_torch_empty()
    # test_dict_identity()
    # test_in_list()
    # test_silce()
    test_dp_layer_rank_mapping()