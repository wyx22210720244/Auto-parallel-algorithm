import pulp
from pyomo.environ import *
from gpu import GPU, Cluster, NODE
from pulp import *
import numpy as np
from collections import defaultdict, Counter
# from training_estimator import get_full_training_memory_consumption, get_args
import json
import copy
from training_estimator import *


# Path: algorithm/main.py

def pp_marginal_utility_homo():
    pass


def pp_marginal_utility_hete():
    pass


def dp_marginal_utility_homo():
    pass


def dp_marginal_utility_hete():
    pass


def tp_marginal_utility_hete():
    pass


def tp_marginal_utility_homo():
    pass


def throughput():
    pass


def greedy(cluster, min_memory_requirement):
    all_gpus = sorted([gpus for nodes in cluster.nodes for gpus in nodes.gpus],
                      key=lambda gpus: gpus.memory,
                      reverse=True)
    # print(f"所有GPU按照显存排序后的结果:",
    #       f"{all_gpus}",sep="\n")
    best_dp_groups = []
    dp_current_groups = []
    dp_current_groups_idx = []
    while all_gpus:
        min_memory_unused = 80
        current_compute_power = 0
        current_memory = 0
        dp_current_groups_idx.append(0)
        dp_current_groups.append(all_gpus[0])
        gpu = all_gpus[0]
        current_memory += gpu.memory
        current_compute_power += gpu.compute_power
        satisfied_gpu_name = None
        satisfied_gpu_idx = None
        for j in range(1, len(all_gpus)):
            if satisfied_gpu_name is not None and satisfied_gpu_name == all_gpus[j].name:
                continue
            current_memory = (sum(gpu.memory for gpu in dp_current_groups)
                              + all_gpus[j].memory)
            current_compute_power = (sum(gpu.compute_power for gpu in dp_current_groups)
                                     + all_gpus[j].compute_power)
            if current_memory >= min_memory_requirement:
                current_memory_unused = current_memory - min_memory_requirement
                if current_memory_unused <= min_memory_unused:
                    min_memory_unused = current_memory_unused
                    satisfied_gpu_name = all_gpus[j].name
                    satisfied_gpu_idx = j
            else:
                dp_current_groups.append(all_gpus[j])
                dp_current_groups_idx.append(j)
        if satisfied_gpu_idx is not None:
            dp_current_groups.append(all_gpus[satisfied_gpu_idx])
            dp_current_groups_idx.append(satisfied_gpu_idx)
            best_dp_groups.append(dp_current_groups)
            for i in range(len(dp_current_groups_idx)):
                all_gpus.pop(dp_current_groups_idx[len(dp_current_groups_idx) - 1 - i])
            dp_current_groups = []
            dp_current_groups_idx = []
        else:
            for i in range(len(dp_current_groups)):
                best_dp_groups[i].append(dp_current_groups[i])
            break
    return best_dp_groups


def optimize(min_memory_usage, all_gpus, dp_num, max_gpu_num_difference, time_limit=20):
    object = LpProblem("DP", LpMaximize)
    minpower = LpVariable("minpower", 0, None, cat=LpInteger)
    min_gpu_num_per_group = LpVariable("min_gpu_num_per_group", 0, None, cat=LpInteger)
    max_gpu_num_per_group = LpVariable("max_gpu_num_per_group", 0, None, cat=LpInteger)
    assignment = LpVariable.dicts("assignment",
                                  (range(len(all_gpus)), range(dp_num)),
                                  0, 1, cat=LpBinary)
    group_power = LpVariable.dicts("group_power", range(dp_num), 0, None, LpInteger)
    # 优化目标
    object += minpower
    # 约束条件
    for i in range(len(all_gpus)):
        object += lpSum([assignment[i][j] for j in range(dp_num)]) == 1
    for j in range(dp_num):
        object += group_power[j] >= minpower
        object += lpSum([assignment[i][j] * all_gpus[i].compute_power for i in range(len(all_gpus))]) == group_power[j]
        object += lpSum([assignment[i][j] * all_gpus[i].memory for i in range(len(all_gpus))]) >= min_memory_usage
        object += lpSum([assignment[i][j] for i in range(len(all_gpus))]) >= min_gpu_num_per_group
        object += lpSum([assignment[i][j] for i in range(len(all_gpus))]) <= max_gpu_num_per_group
    object += (max_gpu_num_per_group - min_gpu_num_per_group) <= max_gpu_num_difference - 1
    # print(object)
    result = object.solve(pulp.PULP_CBC_CMD(msg=True, fracGap=0, options=['-sec', str(time_limit)]))
    # print("Status:", LpStatus[object.status])
    # print("Optimal value:", value(object.objective))
    # print("Optimal var values:")
    # for v in object.variables():
    #     print(v.name, "=", v.varValue)
    # print("Optimal assignment:")
    # for i in range(total_gpus):
    #     for j in range(dp_num):
    #         if assignment[i][j].varValue == 1:
    #             print(f"GPU{i} is assigned to DP{j}")
    # print("Optimal group power:")
    # for j in range(dp_num):
    #     print(f"DP{j} power is {group_power[j].varValue}")
    return result, assignment, group_power


def optimize_test(min_memory_usage, all_gpus, dp_num, max_gpu_num_difference, time_limit=20):
    object = LpProblem("DP", LpMaximize)
    minpower = LpVariable("minpower", 0, None, cat=LpInteger)
    min_gpu_num_per_group = LpVariable("min_gpu_num_per_group", 0, None, cat=LpInteger)
    max_gpu_num_per_group = LpVariable("max_gpu_num_per_group", 0, None, cat=LpInteger)
    assignment = LpVariable.dicts("assignment",
                                  (range(len(all_gpus)), range(dp_num)),
                                  0, 1, cat=LpBinary)
    group_power = LpVariable.dicts("group_power", range(dp_num), 0, None, LpInteger)
    # 优化目标`
    object += minpower
    # 约束条件
    for i in range(len(all_gpus)):
        object += lpSum([assignment[i][j] for j in range(dp_num)]) == 1
    for j in range(dp_num):
        object += group_power[j] >= minpower
        object += lpSum([assignment[i][j] * all_gpus[i].compute_power for i in range(len(all_gpus))]) == group_power[j]
        object += lpSum([assignment[i][j] * all_gpus[i].memory for i in range(len(all_gpus))]) >= min_memory_usage
        object += lpSum([assignment[i][j] for i in range(len(all_gpus))]) >= min_gpu_num_per_group
        object += lpSum([assignment[i][j] for i in range(len(all_gpus))]) <= max_gpu_num_per_group
    object += (max_gpu_num_per_group - min_gpu_num_per_group) <= max_gpu_num_difference - 1
    # print(object)
    result = object.solve(pulp.PULP_CBC_CMD(msg=True, fracGap=0, options=['-sec', str(time_limit)]))
    return result, assignment, group_power, minpower


def search_dp_num(min_memory_usage, all_gpus, max_dp_num, time_limit=5):
    dp_num = max_dp_num
    max_dp_num_difference = 3
    while max_dp_num_difference <= len(all_gpus):
        while dp_num >= 1:
            result, assignment, group_power = optimize(min_memory_usage, all_gpus, dp_num, max_dp_num_difference,
                                                       time_limit=time_limit)
            if result == LpStatusOptimal:
                return dp_num, assignment, group_power, max_dp_num_difference
            else:
                dp_num -= 1
        max_dp_num_difference += 1
    raise ValueError("No solution")


def search_dp_num_test(min_memory_usage, all_gpus, max_dp_num, time_limit=20):
    dp_num = max_dp_num
    max_dp_num_difference = 5
    result, assignment, group_power, minpower = optimize_test(min_memory_usage, all_gpus, dp_num, max_dp_num_difference,
                                                              time_limit=time_limit)
    power_waste = 0
    if result == LpStatusOptimal:
        for j in range(dp_num):
            power_waste += sum(
                assignment[i][j].varValue * all_gpus[i].compute_power for i in range(len(all_gpus))) - minpower.varValue
            print(f"DP{j} power is {group_power[j].varValue}")
        return power_waste
    else:
        return -1


def parallel_3d_strategy(min_memory_usage, all_gpus, max_dp_num, cluster):
    dp_num, assignment, group_power, max_dp_num_difference = search_dp_num(min_memory_usage, all_gpus, max_dp_num)
    dp_allocation, group_quantities = dp_group_gpu_assignment_with_bandwidth(assignment, all_gpus, cluster)
    tp_group = create_tp_group(dp_allocation, cluster, group_quantities)
    pp_group = create_pp_group(dp_allocation, cluster, group_quantities, tp_group)
    return dp_allocation, tp_group, pp_group


def create_tp_group(dp_allocation, cluster):
    tp_group = defaultdict(list)
    tp_group_id = 0
    tp_group_tmp = []
    for group_id, dp_group_gpu in dp_allocation.items():
        for gpu in dp_group_gpu:
            if not tp_group_tmp:
                tp_group_tmp.append(gpu)
                continue
            elif gpu.node == tp_group_tmp[0].node and gpu.name == tp_group_tmp[0].name:
                tp_group_tmp.append(gpu)
                # print(f"tp_group_tmp:{tp_group_tmp}")
                tp_group[tp_group_id] = tp_group_tmp[:]
                # print(f"tp_group:{tp_group}")
                tp_group_id += 1
                del tp_group_tmp[:2]
            else:
                tp_group_tmp.pop()
                tp_group_tmp.append(gpu)
    # print(
    #     f"tp_group:{tp_group}"
    # )
    return tp_group


def create_pp_group(dp_allocation, cluster, tp_group):
    pp_group = defaultdict(list)
    tp_group_id = 0
    pp_group_tmp = []
    for group_id, dp_group_gpu in dp_allocation.items():
        i = 0
        for gpu in dp_group_gpu:
            if tp_group_id < len(tp_group):
                if gpu.node == tp_group[tp_group_id][i].node and gpu.name == tp_group[tp_group_id][i].name:
                    pp_group_tmp.append(gpu)
                    i += 1
                    if len(pp_group_tmp) == 2:
                        pp_group[group_id].append(list(pp_group_tmp))
                        pp_group_tmp = []
                        tp_group_id += 1
                        i = 0
                else:
                    pp_group[group_id].append(gpu)
            else:
                pp_group[group_id].append(gpu)
    return pp_group


def dp_group_gpu_assignment_with_bandwidth(assignments, all_gpus, cluster):
    for idx, inner_dict in assignments.items():
        group_num = len(inner_dict)
        break
    # gpus = ["A100", "V100", "H800"]
    group_quantities = {}
    for group in range(group_num):
        group_quantities[group] = {}
        for gpu in range(len(all_gpus)):
            if assignments[gpu][group].varValue == 1:
                model = all_gpus[gpu].name
                # print(model)
                group_quantities[group][model] = group_quantities[group].get(model, 0) + 1
    print(f"group_quantities:{group_quantities}")
    node_gpu_idx = defaultdict(lambda: defaultdict(list))
    for node in cluster.nodes:
        for gpu in node.gpus:
            node_gpu_idx[node.node_id][gpu.name].append(gpu)
    # print(f"node_gpu_idx:{node_gpu_idx}")
    dp_allocation = defaultdict(list)
    for group_id, group_scheme in group_quantities.items():
        for gpu_model, count in group_scheme.items():
            allocation = 0
            for node_id in list(node_gpu_idx.keys()):
                available_gpu_num = node_gpu_idx[node_id][gpu_model]
                available_gpu_num.sort(key=lambda gpu: gpu.local_rank, reverse=True)
                while available_gpu_num and allocation < count:
                    gpu = available_gpu_num.pop(0)
                    dp_allocation[group_id].append((node_id, gpu))
                    allocation += 1
                if allocation == count:
                    break
    return dp_allocation, group_quantities


def build_cluster(tp_size):
    # cluster
    cluster = Cluster()
    total_nodes = 2
    # gpus_per_node = 8
    a100_num_nodes = 1
    v100_num_nodes = 0
    h800_num_nodes = 1
    a100_gpus_per_node = 8 // tp_size
    v100_gpus_per_node = 8 // tp_size
    h800_gpus_per_node = 8 // tp_size
    # total_gpus = total_nodes * gpus_per_node
    a100_temp = a100_num_nodes
    v100_temp = v100_num_nodes
    h800_temp = h800_num_nodes
    # max_dp_num = 10
    # 构建集群
    for i in range(total_nodes):
        node = NODE(node_id=i)
        if a100_temp > 0:
            for j in range(a100_gpus_per_node):
                a100 = GPU(name='A100', compute_power=100, memory=80, local_rank=j, node=i)
                node.add_gpu(a100)
            a100_temp -= 1
        elif v100_temp > 0:
            for j in range(v100_gpus_per_node):
                v100 = GPU(name='V100', compute_power=50, memory=32, local_rank=j, node=i)
                node.add_gpu(v100)
            v100_temp -= 1
        elif h800_temp > 0:
            for j in range(h800_gpus_per_node):
                h800 = GPU(name='H800', compute_power=185, memory=80, local_rank=j, node=i)
                node.add_gpu(h800)
            h800_temp -= 1
        else:
            break
        cluster.add_node(node)
    cluster.num_gpus = sum(len(node.gpus) for node in cluster.nodes)
    print(f"cluster.num_gpus:{cluster.num_gpus}")
    cluster.num_nodes = len(cluster.nodes)
    print(f"cluster.num_nodes:{cluster.num_nodes}")
    gpu_node_info = defaultdict(list)
    for node in cluster.nodes:
        for gpu in node.gpus:
            gpu_name = gpu.name
        gpu_node_info[node.node_id].append((gpu_name, len(node.gpus)))
    return cluster, gpu_node_info


def get_memory_usage(args):
    memory_usage = get_full_training_memory_consumption(args)
    return memory_usage


def non_linear_optimize(min_memory_usage, cluster, max_dp_num, tp_size):
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
    model.g = Var(range(max_dp_num), within=Reals)
    model.bubble = Var(range(max_dp_num), within=Reals)
    model.m = Var(range(max_dp_num), within=Integers)
    # model.s = Var(range(max_dp_num), within=Reals)
    # model.num_micro_batch = Var(range(max_dp_num), within=NonNegativeReals)
    # model.matrix_q = Var(range(cluster.num_nodes), range(max_dp_num), within=Integers, initialize=0)
    model.d = Var(within=Integers, initialize=1)
    # model.div = Var(range(cluster.num_nodes), range(max_dp_num), within=NonNegativeIntegers)
    # model.tp_num = Var(range(max_dp_num), within=Reals, initialize=0)
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
        model.constraints.add(
            model.m[j] == sum(model.x[i, j] * all_gpus[i].memory * tp_size for i in range(len(all_gpus))))
        model.constraints.add(model.d == sum(model.y[j] for j in range(max_dp_num)))
        model.constraints.add(
            model.g[j] == (sum(model.x[i, j] * all_gpus[i].compute_power for i in range(len(all_gpus))) * (
                    1 - model.bubble[j])))
        model.constraints.add(model.min_dp_group_compute <= model.g[j] + 10000 * (1 - model.y[j]))
        model.constraints.add(model.n[j] >= model.y[j])
        model.constraints.add(model.n[j] <= 1000 * model.y[j])
        model.constraints.add(model.n[j] == sum(model.x[i, j] for i in range(len(all_gpus))))
        # model.constraints.add(model.tp_num[j] == sum(model.div[i, j] for i in range(cluster.num_nodes)))
        # model.constraints.add(model.s[j] == model.n[j]-model.tp_num[j])
        model.constraints.add(model.bubble[j] == (model.n[j] - 1) / (12 - 1 + model.n[j]) * model.y[j])
    model.constraints.add(model.d == sum(model.y[j] for j in range(max_dp_num)))
    for i in range(len(all_gpus)):
        model.constraints.add(sum(model.x[i, j] for j in range(max_dp_num)) == 1)
    # for i in range(cluster.num_nodes):
    #     for j in range(max_dp_num):
    #         model.constraints.add(
    #             model.matrix_q[i, j] == sum(matrix_q[i][k] * model.x[k, j] for k in range(cluster.num_gpus)))
    #         model.constraints.add(model.div[i, j] <= model.matrix_q[i, j] / 2)
    #         model.constraints.add(model.div[i, j] >= model.matrix_q[i, j] / 2 - 1)
    # model.dp_group_bubble_constraint = Constraint(range(max_dp_num), rule=dp_group_bubble_constraint)
    # solver_path = "/Users/wangyuxiao/Downloads/Bonmin-1.8.8/build/bin/bonmin"
    # solver = SolverFactory('bonmin', executable=solver_path)
    solver = SolverFactory('scip')
    solver.options['limits/time'] = 300
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
        dp_allocation = defaultdict(list)
        for dp_group in range(max_dp_num):
            for gpu in range(len(all_gpus)):
                if model.y[dp_group].value > 0.5 and model.x[gpu, dp_group].value > 0.5:
                    dp_allocation[dp_group].append(all_gpus[gpu])
        dp_allocation_reorded = defaultdict(list)
        idx = 0
        for dp_group, gpus in dp_allocation.items():
            dp_allocation_reorded[idx] = gpus
            idx += 1
        print(f"dp_allcation:{dp_allocation_reorded}")
        # tp_group = create_tp_group(dp_allcation, cluster)
        # print(f"tp_group:{tp_group}")
        # pp_group = create_pp_group(dp_allcation, cluster, tp_group)
        # print(f"pp_group:{pp_group}")
        return count_dp_gpu(dp_allocation_reorded), dp_allocation_reorded
    elif result.solver.termination_condition == TerminationCondition.infeasible:
        assert "Model is infeasible!"
    else:
        assert "Solver terminated with condition:", result.solver.termination_condition


def non_linear_optimize_fixed_tp(min_memory_usage, cluster, max_dp_num):
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
    model.d = Var(within=Integers, initialize=1)
    # 目标函数
    model.object = Objective(expr=model.min_dp_group_compute * model.d, sense=maximize)
    # constraint
    model.constraints = ConstraintList()
    for j in range(max_dp_num):
        # 显存约束
        model.constraints.add(model.m[j] * model.y[j] + 10000 * (1 - model.y[j]) >= min_memory_usage)
        model.constraints.add(
            model.m[j] == sum(model.x[i, j] * all_gpus[i].memory for i in range(len(all_gpus))))
        # 每个DP组内的总算力
        model.constraints.add(
            model.g[j] == (sum(model.x[i, j] * all_gpus[i].compute_power for i in range(len(all_gpus))) * (
                    1 - model.bubble[j])))
        # 取出所有DP组内的最小有效算力
        model.constraints.add(model.min_dp_group_compute <= model.g[j] + 10000 * (1 - model.y[j]))
        # 指示函数转化，每个DP组内是否有gpu
        model.constraints.add(model.n[j] >= model.y[j])
        model.constraints.add(model.n[j] <= 1000 * model.y[j])
        # 每个DP组的卡数
        model.constraints.add(model.n[j] == sum(model.x[i, j] for i in range(len(all_gpus))))
        # DP组内的stage数量
        # model.constraints.add(model.s[j] == model.n[j])
        # PP组的bubble计算
        model.constraints.add(model.bubble[j] == (model.n[j] - 1) / (8 - 1 + model.n[j]) * model.y[j])
    # DP组的数量
    model.constraints.add(model.d == sum(model.y[j] for j in range(max_dp_num)))
    # 每个gpu只能分配到一个DP组
    for i in range(len(all_gpus)):
        model.constraints.add(sum(model.x[i, j] for j in range(max_dp_num)) == 1)
    solver_path = "/Users/wangyuxiao/Downloads/Bonmin-1.8.8/build/bin/bonmin"
    # solver = SolverFactory('bonmin', executable=solver_path)
    solver = SolverFactory('scip')
    solver.options['limits/time'] = 600
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
        dp_allocation = defaultdict(list)
        for dp_group in range(max_dp_num):
            for gpu in range(len(all_gpus)):
                if model.y[dp_group].value > 0.5 and model.x[gpu, dp_group].value > 0.5:
                    dp_allocation[dp_group].append(all_gpus[gpu])
        dp_allocation_reorded = defaultdict(list)
        idx = 0
        for dp_group, gpus in dp_allocation.items():
            dp_allocation_reorded[idx] = gpus
            idx += 1
        print(f"dp_allcation:{dp_allocation_reorded}")
        # tp_group = create_tp_group(dp_allcation, cluster)
        # print(f"tp_group:{tp_group}")
        # pp_group = create_pp_group(dp_allcation, cluster, tp_group)
        # print(f"pp_group:{pp_group}")
        return count_dp_gpu(dp_allocation_reorded), dp_allocation_reorded
    elif result.solver.termination_condition == TerminationCondition.infeasible:
        assert "Model is infeasible!"
    else:
        assert "Solver terminated with condition:", result.solver.termination_condition


def assign_pp_layers(dp_allocation, cluster, model_layers, layer_storage, tp_size):
    all_gpus = []
    for nodes in cluster.nodes:
        for gpus in nodes.gpus:
            all_gpus.append(gpus)
    layer_allocation = defaultdict(list)
    dp_idx = 0
    # layer_storage = 10
    for idx, pp_group in dp_allocation.items():
        pp_group_compute = []
        pp_group_memory = []
        for i in range(len(pp_group)):
            pp_group_compute.append(pp_group[i].compute_power)
            pp_group_memory.append(pp_group[i].memory)
        model = ConcreteModel()
        # 每个DP组的stage数
        num_stage = len(pp_group_compute)
        model.l = Var(range(num_stage), domain=NonNegativeIntegers)
        # model.g = Var(range(num_stage),domain=NonNegativeReals)
        model.max_allocation_ratio = Var(domain=NonNegativeReals)
        # 目标
        model.object = Objective(expr=model.max_allocation_ratio, sense=minimize)
        # 约束
        model.constraints = ConstraintList()
        for i in range(num_stage):
            model.constraints.add(model.max_allocation_ratio >= (pp_group_compute[i] / model.l[i]))
            model.constraints.add(model.l[i] * layer_storage <= pp_group_memory[i] * tp_size)
        model.constraints.add(model_layers == sum(model.l[i] for i in range(num_stage)))
        solver = SolverFactory('scip')
        solver.options['limits/time'] = 200
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
            for i in range(num_stage):
                optimal_l = model.l[i].value
                layer_allocation[dp_idx].append(optimal_l)
            dp_idx += 1
        elif result.solver.termination_condition == TerminationCondition.infeasible:
            assert "Model is infeasible!"
        else:
            assert "Solver terminated with condition:", result.solver.termination_condition
    return layer_allocation


def count_dp_gpu(dp_allocation):
    dp_gpu_count = defaultdict(Counter)
    idx = 0
    for _, gpus in dp_allocation.items():
        for gpu in gpus:
            dp_gpu_count[idx][gpu.name] += 1
        idx += 1
    return dp_gpu_count


def reorganize_idx(dp_allocation, pp_stage_info, layer_allocation):
    dp_allocation_reorganize = copy.deepcopy(dp_allocation)
    layer_allocation_reorganize = copy.deepcopy(layer_allocation)
    for dp_group, gpus in dp_allocation.items():
        for idx, gpu in enumerate(gpus):
            dp_allocation_reorganize[dp_group][pp_stage_info[dp_group][idx]] = gpus[idx]
            layer_allocation_reorganize[dp_group][pp_stage_info[dp_group][idx]] = layer_allocation[dp_group][idx]
    for key in layer_allocation_reorganize:
        layer_allocation_reorganize[key] = [round(value, 1) for value in layer_allocation_reorganize[key]]
    return dp_allocation_reorganize, layer_allocation_reorganize


def calculate_memory_usage(args, num_layer, stage, max_stage):
    args.num_layers = num_layer
    # 获取所拥有的层的前向激活内存占用
    activation_memory = (get_activation_memory(args))
    # 获取各stage的num of mb 的内存占用
    if args.num_micro_batches > max_stage:
        args.num_micro_batches = max_stage
    activation_memory = activation_memory * (args.num_micro_batches - stage)
    # 获取模型固定内存占用
    fixed_memory = get_memory_without_activation(args)
    return bytes_to_gb(fixed_memory + activation_memory)


def assign_pp_stage(dp_allocation, cluster, args, layer_allocation):
    # 为每种型号的gpu分配gloabl node id,表明每个节点上的gpu型号和gpu数量
    gpu_node_info = defaultdict(list)
    for node in cluster.nodes:
        for gpu in node.gpus:
            gpu_name = gpu.name
        gpu_node_info[node.node_id].append(gpu_name)
        gpu_node_info[node.node_id].append(len(node.gpus))
    gpu_compute_power = ["A100", "H800", "V100"]
    dp_gpu_count = count_dp_gpu(dp_allocation)
    # 初始化node、stage信息
    pp_stage_info = defaultdict(list)
    for dp_group, gpus in dp_allocation.items():
        for i in range(len(gpus)):
            pp_stage_info[dp_group].append(None)
    dp_num = len(dp_allocation)
    # 从算力最低的型号开始往下判断
    k = 0
    stage = 0
    label = 0
    assigned_stage = defaultdict(list)
    # 选出所有DP组中能处于同一节点上的stage
    while k < len(gpu_compute_power):
        # 取出gpu型号
        gpu_model = gpu_compute_power[k]
        if all(counter[gpu_model] >= 1 for counter in dp_gpu_count.values()):
            for node_id, gpu_info in gpu_node_info.items():
                if gpu_info[0] == gpu_model and gpu_info[1] >= dp_num:
                    for counter in dp_gpu_count.values():
                        counter[gpu_model] -= 1
                    label = 1
                    node = node_id
                    for dp_group, gpus in dp_allocation.items():
                        for idx, gpu in enumerate(gpus):
                            if gpu.name == gpu_model and pp_stage_info[dp_group][idx] is None:
                                while stage < len(gpus):
                                    if stage not in assigned_stage[dp_group]:
                                        gpu.node = node
                                        pp_stage_info[dp_group][idx] = stage
                                        assigned_stage[dp_group].append(stage)
                                        break
                                    else:
                                        stage += 1
                                stage = 0
                                break
                    gpu_info[1] -= dp_num
                    break
        if label == 0:
            k += 1
        label = 0
    stage = 0
    k = 0
    # 处理剩余stage
    while k < len(gpu_compute_power):
        gpu_model = gpu_compute_power[k]
        for dp_group, gpus in dp_allocation.items():
            for idx, gpu in enumerate(gpus):
                if gpu.name == gpu_model and pp_stage_info[dp_group][idx] is None:
                    for node_id, gpu_info in gpu_node_info.items():
                        if gpu_info[0] == gpu_model and gpu_info[1] >= 1:
                            gpu_info[1] -= 1
                            while stage < len(gpus):
                                if stage not in assigned_stage[dp_group]:
                                    gpu.node = node_id
                                    pp_stage_info[dp_group][idx] = stage
                                    assigned_stage[dp_group].append(stage)
                                    break
                                else:
                                    stage += 1
                            stage = 0
                            break
        k += 1
        local_rank = 0
        for i in range(cluster.num_nodes):
            for dp_group, gpus in dp_allocation.items():
                for idx, gpu in enumerate(gpus):
                    if gpu.node == i:
                        gpu.local_rank = local_rank
                        local_rank += 1
            local_rank = 0
    return dp_allocation, pp_stage_info


def dict_to_json(dp_allocation, layer_allocation, tp_size):
    result = defaultdict(list)
    if tp_size == 2:
        for key, value in dp_allocation.items():
            for gpu in value:
                result[key].append([gpu.node * 8 + gpu.local_rank, gpu.node * 8 + gpu.local_rank + 4])
        with open("allocations.json", "w") as f:
            json.dump(result, f)
        for key, value in layer_allocation.items():
            sum = 0
            for idx, layer in enumerate(value):
                value[idx] += sum
                sum += layer
        with open("layers.json", "w") as f:
            json.dump(layer_allocation, f)
    if tp_size == 4:
        for key, value in dp_allocation.items():
            for gpu in value:
                result[key].append([gpu.node * 8 + gpu.local_rank, gpu.node * 8 + gpu.local_rank + 2,
                                    gpu.node * 8 + gpu.local_rank + 4, gpu.node * 8 + gpu.local_rank + 6])
        with open("allocations.json", "w") as f:
            json.dump(result, f)
        for key, value in layer_allocation.items():
            sum = 0
            for idx, layer in enumerate(value):
                value[idx] += sum
                sum += layer
        with open("layers.json", "w") as f:
            json.dump(layer_allocation, f)
    if tp_size == 8:
        for key, value in dp_allocation.items():
            for gpu in value:
                result[key].append([gpu.node * 8 + gpu.local_rank, gpu.node * 8 + gpu.local_rank + 1,
                                    gpu.node * 8 + gpu.local_rank + 2, gpu.node * 8 + gpu.local_rank + 3,
                                    gpu.node * 8 + gpu.local_rank + 4, gpu.node * 8 + gpu.local_rank + 5,
                                    gpu.node * 8 + gpu.local_rank + 6, gpu.node * 8 + gpu.local_rank + 7])
        with open("allocations.json", "w") as f:
            json.dump(result, f)
        for key, value in layer_allocation.items():
            sum = 0
            for idx, layer in enumerate(value):
                value[idx] += sum
                sum += layer
        with open("layers.json", "w") as f:
            json.dump(layer_allocation, f)


def assign_dp_group():
    layer_allocation = json.load(open("layers.json"))
    dp_allocation = json.load(open("allocations.json"))
    unique_layer_allocation = set()
    # unique_layer_allocation.add(0)
    for _, layers in layer_allocation.items():
        for layer in layers:
            unique_layer_allocation.add(layer)
    sorted_unique_intervals = sorted(list(unique_layer_allocation))
    print(f"sorted_unique_intervals:{sorted_unique_intervals}")
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
    for key, _ in dp_group_comm_up.items():
        dp_group_comm[key].append(dp_group_comm_up[key])
        dp_group_comm[key].append(dp_group_comm_down[key])
    print(f"dp_group_comm_up:{dp_group_comm_up}")
    print(f"dp_group_comm_down:{dp_group_comm_down}")
    print(dp_group_comm)


# 判断当前gpu在的pp stage中是否oom
def is_out_of_memory(gpu, stage, max_stage, args, layers):
    t = args.tensor_model_parallel_size
    max_memory = gpu.memory * t
    used_memory = get_stage_memory(stage, args, max_stage, layers)
    return used_memory > max_memory


# 更新重新分配后的load_ratio
def update_valid_load_ratio(dp_allocation, layer_allocation,valid_gpus,dp_group):
    valid_load_ratio = defaultdict(list)
    for dp_group,gpus in valid_gpus.items():
        for stage in gpus:
            layer = layer_allocation[dp_group][stage]
            gpu = dp_allocation[dp_group][stage]
            valid_load_ratio[dp_group].append(layer/gpu.compute_power)
    return sort_valid_gpus(valid_gpus,valid_load_ratio)


# 按照load_ratio排序valid_gpus
def sort_valid_gpus(valid_gpus, valid_load_ratio):
    sorted_valid_gpus = defaultdict(list)
    for dp_group, gpus in valid_gpus.items():
        load_ratio = valid_load_ratio[dp_group]
        pair = zip(gpus, load_ratio)
        sorted_pair = sorted(pair, key=lambda x: x[1])
        sorted_valid_gpus[dp_group] = [items[0] for items in sorted_pair]
    return sorted_valid_gpus


def transfer_layer(dp_allocation, layer_allocation, oom_gpu, valid_gpu, dp_group, max_stage, args):
    oom_layer = layer_allocation[dp_group][oom_gpu]
    vaild_layer = layer_allocation[dp_group][valid_gpu]
    oom_layer -= 1
    vaild_layer += 1
    if not is_out_of_memory(dp_allocation[dp_group][valid_gpu], valid_gpu, max_stage, args, vaild_layer):
        layer_allocation[dp_group][oom_gpu] = oom_layer
        layer_allocation[dp_group][valid_gpu] = vaild_layer
        return True
    else:
        return False


# 保证显存均衡，每个stage都不会oom
def load_balance(dp_allocation, layer_allocation, args):
    # oom的gpu集合(stage号)
    oom_gpus = defaultdict(list)
    # 没有显存溢出的gpu集合(stage号)
    valid_gpus = defaultdict(list)
    valid_load_ratio = defaultdict(list)
    for dp_group, gpus in dp_allocation.items():
        for stage, gpu in enumerate(gpus):
            layer = layer_allocation[dp_group][stage]
            if is_out_of_memory(gpu, stage, len(gpus), args, layer):
                oom_gpus[dp_group].append(stage)
            else:
                valid_gpus[dp_group].append(stage)
                valid_load_ratio[dp_group].append(layer / gpu.compute_power)
    # 把valid_gpu按照负载比例升序排序，优先shift到load_ratio低的stage上
    sorted_valid_gpus = sort_valid_gpus(valid_gpus, valid_load_ratio)
    # 转移工作负载
    for dp_group, gpus in dp_allocation.items():
        max_stage = len(gpus)
        while len(oom_gpus[dp_group]) > 0:
            assert len(valid_gpus[dp_group])>0; "Invalid gpus"
            oom_gpu = oom_gpus[dp_group][0]
            valid_gpu = sorted_valid_gpus[dp_group][0]
            if transfer_layer(dp_allocation, layer_allocation, oom_gpu, valid_gpu, dp_group, max_stage, args):
                sorted_valid_gpus = update_valid_load_ratio(dp_allocation,layer_allocation,valid_gpus,dp_group)
                if not is_out_of_memory(gpus[oom_gpu], oom_gpu, max_stage, args, layer_allocation[dp_group][oom_gpu]):
                    oom_gpus[dp_group].pop()
            else:
                valid_gpus[dp_group].pop()


if __name__ == '__main__':
    # cluster
    args = get_args()
    # args.tensor_model_parallel_size = 2
    cluster, gpu_node_info = build_cluster(args.tensor_model_parallel_size)
    max_dp_num = 5
    # 任务信息
    # args.num_layers = 24
    # args.global_batch_size = 320
    # args.micro_batch_size = 16
    # args.hidden_size = 4096
    # args.num_attention_heads = 40
    # args.seq_length = 1024
    min_memory_usage = get_memory(args)
    print(f"min_memory_usage:{min_memory_usage}")
    all_gpus = sorted([gpus for nodes in cluster.nodes for gpus in nodes.gpus],
                      key=lambda gpus: gpus.compute_power,
                      reverse=True)
    print(all_gpus)
    # pp_group = non_linear_optimize(min_memory_usage, cluster, max_dp_num)
    # pp_group_json = dict_to_json(pp_group)
    # print(f"pp_group_json:{pp_group_json}")
    dp_gpu_count, dp_allocation = non_linear_optimize(min_memory_usage, cluster, max_dp_num, args.tensor_model_parallel_size)
    num_dp_group = len(dp_allocation)
    args.num_micro_batches = args.global_batch_size // args.micro_batch_size // num_dp_group
    layer_allocation = assign_pp_layers(dp_allocation, cluster, args.num_layers, 1, args.tensor_model_parallel_size)
    dp_allocation, pp_stage_info = assign_pp_stage(dp_allocation, cluster, args, layer_allocation)
    print(f"dp_allocation:{dp_allocation}")
    print(f"layer_allocation:{layer_allocation}")
    print(f"pp_stage_info:{pp_stage_info}")
    # 下述模块开始做memory_constrainted_load_balance
    load_balance(dp_allocation, layer_allocation, args)
    dp_allocation, layer_allocation = reorganize_idx(dp_allocation, pp_stage_info, layer_allocation)
    dict_to_json(dp_allocation, layer_allocation, args.tensor_model_parallel_size)
    print(f"dp_allocation:{dp_allocation}")
    print(f"layer_allocation:{layer_allocation}")
    assign_dp_group()
