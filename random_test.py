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
    maxlength = 1
    for value in list_var:
        if len(value) >= maxlength:
            maxlength = len(value)
    if maxlength > 1:
        for value in list_var:
            if len(value) <= 1:
                value.append(value[0])
    return list_var


def test_torch_empty():
    tensor = torch.randn(10, 5)
    print(tensor)
    print(tensor[:5, :])


def test_dict_identity():
    dict = {'a': 1, 'b': 1}
    print(dict['a'] == dict['b'])


def test_in_list():
    list1 = [1, 2, 3, 4]
    print(4 in list1)


def test_silce():
    tensor = torch.randn(10, 5)
    tensor2 = tensor[:5, :]
    print(tensor)
    print(tensor2)
    tensor2[0, 0] = 0
    print(tensor)


def test_dp_layer_rank_mapping():
    map = {"1": {0: [1, 2], 1: [3, 4]}, "2": {2: [1, 4]}}
    filename = "dp_layer_rank_mapping.json"
    with open(filename, 'w') as f:
        json.dump(map, f)
    # result = json.dumps(map)


def get_layer_rank_mapping():
    """Return layer rank mapping"""
    with open("dp_layer_rank_mapping.json", 'r') as f:
        layer_rank_mapping = json.load(f)
    return layer_rank_mapping


def find_matching_layer_dp_group():
    # args = get_args()
    layer_rank_mapping = get_layer_rank_mapping()
    num_layer = 6
    # dp_group = list(layer_rank_mapping.keys())
    matching_layer = {}
    for i in range(1, num_layer + 1):
        matching_layer[i] = []
    layer_curr = 1
    for dp_num, rank_layer in layer_rank_mapping.items():
        for rank, layer in rank_layer.items():
            while True:
                if layer_curr <= layer[1]:
                    matching_layer[layer_curr].append(rank)
                    layer_curr += 1
                else:
                    break
        layer_curr = 1
    print(matching_layer)
    return matching_layer


def test_tensor_view():
    tensor1 = torch.Tensor()
    tensor2 = torch.tensor([1, 2, 3])
    tensor1 = torch.cat((tensor1, tensor2), dim=0)
    print(tensor1)


def test_name():
    list1 = {0: [("H800", 4)]}
    gpu_compute_power = ["H800", "A100", "V100"]
    print(list1[0][0][0])

def assign_dp_group():
    layer_allocation = json.load(open("layers.json"))
    dp_allocation = json.load(open("allocations.json"))
    for key, value in dp_allocation.items():
        for gpus in value:
            tp_size = len(gpus)
    unique_layer_allocation = set()
    # unique_layer_allocation.add(0)
    for _, layers in layer_allocation.items():
        for layer in layers:
            unique_layer_allocation.add(layer)
    sorted_unique_intervals = sorted(list(unique_layer_allocation))
    print(f"sorted_unique_intervals:{sorted_unique_intervals}")
    if tp_size ==2:
        dp_group_comm_up = defaultdict(list)
        dp_group_comm_down = defaultdict(list)
        group_idx = 0
        for layer in sorted_unique_intervals:
            for dp_group, gpus in dp_allocation.items():
                for idx, gpu in enumerate(gpus):
                    if layer_allocation[dp_group][idx] >= layer:
                        dp_group_comm_up[group_idx].append(gpu[0])
                        dp_group_comm_down[group_idx].append(gpu[1])
                        break
            group_idx += 1
        dp_group_comm = defaultdict(list)
        for key,_ in dp_group_comm_up.items():
            dp_group_comm[key].append(dp_group_comm_up[key])
            dp_group_comm[key].append(dp_group_comm_down[key])
        print(f"dp_group_comm_up:{dp_group_comm_up}")
        print(f"dp_group_comm_down:{dp_group_comm_down}")
        print(dp_group_comm)
    if tp_size == 4:
        dp_group_comm_1 = defaultdict(list)
        dp_group_comm_2 = defaultdict(list)
        dp_group_comm_3 = defaultdict(list)
        dp_group_comm_4 = defaultdict(list)
        group_idx = 0
        for layer in sorted_unique_intervals:
            for dp_group, gpus in dp_allocation.items():
                for idx, gpu in enumerate(gpus):
                    if layer_allocation[dp_group][idx] >= layer:
                        dp_group_comm_1[group_idx].append(gpu[0])
                        dp_group_comm_2[group_idx].append(gpu[1])
                        dp_group_comm_3[group_idx].append(gpu[2])
                        dp_group_comm_4[group_idx].append(gpu[3])
                        break
            group_idx += 1
        dp_group_comm = defaultdict(list)
        for key, _ in dp_group_comm_1.items():
            dp_group_comm[key].append(dp_group_comm_1[key])
            dp_group_comm[key].append(dp_group_comm_2[key])
            dp_group_comm[key].append(dp_group_comm_3[key])
            dp_group_comm[key].append(dp_group_comm_4[key])
        print(f"dp_group_comm_1:{dp_group_comm_1}")
        print(f"dp_group_comm_2:{dp_group_comm_2}")
        print(f"dp_group_comm_3:{dp_group_comm_3}")
        print(f"dp_group_comm_4:{dp_group_comm_4}")
        print(dp_group_comm)
        
    dp_layer_comm = []
    sorted_unique_intervals.insert(0, 0)
    for idx in range(1,len(sorted_unique_intervals)):
        dp_layer_comm.append(sorted_unique_intervals[idx]-sorted_unique_intervals[idx-1])
    print(
        f"dp_layer_comm:{dp_layer_comm}"
    )
def test_dp_count():
    dp_allocation = json.load(open("allocations.json"))
    gpu_count  = 0
    for key, value in dp_allocation.items():
        for gpu in value:
            if isinstance(gpu, list):
                gpu_count += len(gpu)
            else:
                gpu_count += 1
    print(gpu_count)

def non_linear_optimize(min_memory_usage, cluster, max_dp_num):
    # all_gpus = sorted([gpus for nodes in cluster.nodes for gpus in nodes.gpus],
    #                   key=lambda gpus: gpus.compute_power,
    #                    reverse=True)
    all_gpus = []
    for nodes in cluster.nodes:
        for gpus in nodes.gpus:
            all_gpus.append(gpus)
    print(f"all_gpus:{all_gpus}")
    # model
    model = ConcreteModel()
    model.x = Var(range(len(all_gpus)), range(max_dp_num), within=Binary)
    model.y = Var(range(max_dp_num), within=Binary)
    model.min_dp_group_compute = Var(within=Integers, initialize=0)
    model.n = Var(range(max_dp_num), within=NonNegativeReals)
    model.g = Var(range(max_dp_num), within=Integers)
    model.bubble = Var(range(max_dp_num), within=Reals)
    model.m = Var(range(max_dp_num), within=Integers)
    model.s = Var(range(max_dp_num), within=Integers)
    # model.num_micro_batch = Var(range(max_dp_num), within=NonNegativeReals)
    model.matrix_q = Var(range(cluster.num_nodes), range(max_dp_num), within=Integers, initialize=0)
    model.d = Var(within=Integers, initialize=1)
    model.div = Var(range(cluster.num_nodes), range(max_dp_num), within=NonNegativeIntegers)
    model.tp_num = Var(range(max_dp_num), within=Reals, initialize=0)
    matrix_q = [[0 for _ in range(cluster.num_gpus)] for _ in range(cluster.num_nodes)]

    all_gpus = []
    idx = 0
    for nodes in cluster.nodes:
        for gpus in nodes.gpus:
            all_gpus.append(gpus)
            matrix_q[nodes.node_id][idx] = 1
            idx += 1
    print(f"all_gpus:{all_gpus}")
    print(f"matrix_N:{matrix_q}")

    model.object = Objective(expr=model.min_dp_group_compute * model.d, sense=maximize)
    # constraint
    model.constraints = ConstraintList()
    # memory
    for j in range(max_dp_num):
        model.constraints.add(model.m[j] * model.y[j] + 10000 * (1 - model.y[j]) >= min_memory_usage)
        model.constraints.add(model.m[j] == sum(model.x[i, j] * all_gpus[i].memory for i in range(len(all_gpus))))
        model.constraints.add(model.d == sum(model.y[j] for j in range(max_dp_num)))
        model.constraints.add(
            model.g[j] == (sum(model.x[i, j] * all_gpus[i].compute_power for i in range(len(all_gpus))) * (
                    1 - model.bubble[j])))
        model.constraints.add(model.min_dp_group_compute <= model.g[j] + 10000 * (1 - model.y[j]))
        model.constraints.add(model.n[j] >= model.y[j])
        model.constraints.add(model.n[j] <= 1000 * model.y[j])
        model.constraints.add(model.n[j] == sum(model.x[i, j] for i in range(len(all_gpus))))
        model.constraints.add(model.tp_num[j] == sum(model.div[i, j] for i in range(cluster.num_nodes)))
        model.constraints.add(model.s[j] == model.n[j] - model.tp_num[j])
        model.constraints.add(model.bubble[j] == (model.s[j] - 1) / (4 - 1 + model.s[j]) * model.y[j])
    model.constraints.add(model.d == sum(model.y[j] for j in range(max_dp_num)))
    for i in range(len(all_gpus)):
        model.constraints.add(sum(model.x[i, j] for j in range(max_dp_num)) == 1)
    for i in range(cluster.num_nodes):
        for j in range(max_dp_num):
            model.constraints.add(
                model.matrix_q[i, j] == sum(matrix_q[i][k] * model.x[k, j] for k in range(cluster.num_gpus)))
            model.constraints.add(model.div[i, j] <= model.matrix_q[i, j] / 2)
            model.constraints.add(model.div[i, j] >= model.matrix_q[i, j] / 2 - 1)
    # model.dp_group_bubble_constraint = Constraint(range(max_dp_num), rule=dp_group_bubble_constraint)
    solver_path = "/Users/wangyuxiao/Downloads/Bonmin-1.8.8/build/bin/bonmin"
    # solver = SolverFactory('bonmin', executable=solver_path)
    solver = SolverFactory('scip')
    solver.options['limits/time'] = 600
    # solver.options['max_iter'] = 1000
    result = solver.solve(model, tee=True)
    # 检查求解器是否找到了一个最优解
    # 打印变量 x 的值
    print("Solver Status:", result.solver.status)
    print("Solver Termination Condition:", result.solver.termination_condition)

    # 判断求解是否成功
    if result.solver.termination_condition == TerminationCondition.optimal or result.solver.termination_condition == TerminationCondition.maxTimeLimit:
        print("Solution is optimal!")
        # 打印变量的值
        print(f"目标函数的值为：{model.object()}")
        model_variables = model.component_objects(Var)
        for var_obj in model_variables:
            print(f"Variable {var_obj.name}:")
            for index in var_obj:
                print(f"  {var_obj.name}[{index}] = {var_obj[index].value}")
        dp_allcation = defaultdict(list)
        for dp_group in range(max_dp_num):
            for gpu in range(len(all_gpus)):
                if model.y[dp_group].value > 0.5 and model.x[gpu, dp_group].value > 0.5:
                    dp_allcation[dp_group].append(all_gpus[gpu])
        print(f"dp_allcation:{dp_allcation}")
        tp_group = create_tp_group(dp_allcation, cluster)
        print(f"tp_group:{tp_group}")
        pp_group = create_pp_group(dp_allcation, cluster, tp_group)
        print(f"pp_group:{pp_group}")
    elif result.solver.termination_condition == TerminationCondition.infeasible:
        assert "Model is infeasible!"
    else:
        assert "Solver terminated with condition:", result.solver.termination_condition
    return pp_group

def test_pop():
    x = [1,2,3,4]
    y = x.pop(0)
    print(y)
if __name__ == '__main__':
#     # test_origin_group_allocation()
#     # test_torch_empty()
#     # test_dict_identity()
#     # test_in_list()
#     # test_silce()
#     # test_dp_layer_rank_mapping()
#     # test_name(
# 
#     assign_dp_group(
#     )
#     test_dp_count()
    test_pop()
    
