import pulp
from pyomo.environ import *
from gpu import GPU, Cluster, NODE
from pulp import *
import numpy as np
from collections import defaultdict
from training_estimator import get_full_training_memory_consumption, get_args


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


def build_cluster():
    # cluster
    cluster = Cluster()
    total_nodes = 6
    # gpus_per_node = 8
    a100_num_nodes = 3
    v100_num_nodes = 2
    h800_num_nodes = 1
    a100_gpus_per_node = 8
    v100_gpus_per_node = 8
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
                a100 = GPU(name='A100', compute_power=100, memory=50, local_rank=j, node=i)
                node.add_gpu(a100)
            a100_temp -= 1
        elif v100_temp > 0:
            for j in range(v100_gpus_per_node):
                v100 = GPU(name='V100', compute_power=30, memory=32, local_rank=j, node=i)
                node.add_gpu(v100)
            v100_temp -= 1
        elif h800_temp > 0:
            for j in range(v100_gpus_per_node):
                h800 = GPU(name='H800', compute_power=200, memory=80, local_rank=j, node=i)
                node.add_gpu(h800)
            h800_temp -= 1
        else:
            break
        cluster.add_node(node)
    cluster.num_gpus = sum(len(node.gpus) for node in cluster.nodes)
    print(f"cluster.num_gpus:{cluster.num_gpus}")
    cluster.num_nodes = len(cluster.nodes)
    print(f"cluster.num_nodes:{cluster.num_nodes}")
    return cluster


def get_memory_usage(args):
    memory_usage = get_full_training_memory_consumption(args)
    return memory_usage


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


def non_linear_optimize_test(min_memory_usage, all_gpus, max_dp_num):
    model = ConcreteModel()
    model.minpower = Var(domain=NonNegativeIntegers, initialize=0)
    model.assignment = Var(range(len(all_gpus)), range(max_dp_num), domain=Binary)
    model.group_power = Var(range(max_dp_num), domain=NonNegativeIntegers)

    model.objective = Objective(expr=model.minpower, sense=maximize)
    model.constraints = ConstraintList()
    for i in range(len(all_gpus)):
        model.constraints.add(sum(model.assignment[i, j] for j in range(max_dp_num)) == 1)
    for j in range(max_dp_num):
        model.constraints.add(model.group_power[j] >= model.minpower)
        model.constraints.add(
            sum(model.assignment[i, j] * all_gpus[i].compute_power for i in range(len(all_gpus))) == model.group_power[
                j])
        model.constraints.add(
            sum(model.assignment[i, j] * all_gpus[i].memory for i in range(len(all_gpus))) >= min_memory_usage)

    solver_path = "/Users/wangyuxiao/Downloads/Bonmin-1.8.8/build/bin/bonmin"
    solver = SolverFactory('bonmin', executable=solver_path)
    result = solver.solve(model, tee=True)
    print("Solver Status:", result.solver.status)
    print("Solver Termination Condition:", result.solver.termination_condition)

    # 判断求解是否成功
    if result.solver.termination_condition == TerminationCondition.optimal:
        print("Solution is optimal!")
    elif result.solver.termination_condition == TerminationCondition.infeasible:
        print("Model is infeasible!")
    else:
        print("Solver terminated with condition:", result.solver.termination_condition)

    # 打印变量的值
    model_variables = model.component_objects(Var)
    for var_obj in model_variables:
        print(f"Variable {var_obj.name}:")
        for index in var_obj:
            print(f"  {var_obj.name}[{index}] = {var_obj[index].value}")


if __name__ == '__main__':
    # cluster
    cluster = build_cluster()
    print(cluster)
    max_dp_num = 10
    # args = get_args()
    # 任务信息
    min_memory_usage = 1000
    # print(f"min_memory_usage:{min_memory_usage}")
    all_gpus = sorted([gpus for nodes in cluster.nodes for gpus in nodes.gpus],
                      key=lambda gpus: gpus.compute_power,
                      reverse=True)
    print(all_gpus)
    # dp_group, tp_group, pp_group = parallel_3d_strategy(min_memory_usage, all_gpus, max_dp_num, cluster)
    # # print(f"dp_group:{dp_group}\n"
    # #       f"tp_group:{tp_group}\n"
    # #       f"pp_group:{pp_group}")
    # # for group_id, gpu in dp_group.items():
    # #     print(f"第{group_id + 1}组DP组所分配的GPU为:\n"
    # #           f"{gpu}")
    # # print("=====================================")
    # # for group_id, gpu in tp_group.items():
    # #     print(f"第{group_id + 1}组TP组所分配的GPU为:\n"
    # #           f"{gpu}")
    # # print("=====================================")
    # for group_id, gpu in pp_group.items():
    #     print(f"第{group_id + 1}组DP组所分配的总的GPU为:\n"
    #           f"{gpu}")
    #     print(f"各PP stage的具体分配情况如下:")
    #     for idx, tuple in enumerate(gpu):
    #         print(f"第{idx}个stage所分配的GPU为:\n"
    #               f"{tuple}")

    # result = []
    # for i in range(max_dp_num):
    #     power_waste = search_dp_num_test(min_memory_usage, all_gpus, i)
    #     result.append(power_waste)
    #     print(f"dp_num is {i},power_waste{power_waste}")
    # dp_num, assignment, group_power, max_dp_num_difference = search_dp_num(min_memory_usage, all_gpus, max_dp_num)
    # print(f"Optimal assignment:{assignment}")
    # for i in range(total_gpus):
    #     for j in range(dp_num):
    #         if assignment[i][j].varValue == 1:
    #             print(f"GPU{i} is assigned to DP{j}")
    # print("Optimal group power:")
    # for j in range(dp_num):
    #     print(f"DP{j} power is {group_power[j].varValue}")
    # print("Optimal group memory:")
    # for j in range(dp_num):
    #     print(f"DP{j} memory is {sum([assignment[i][j].varValue * all_gpus[i].memory for i in range(total_gpus)])}")
    # print("DP num difference is:\n", max_dp_num_difference - 1)
    # print("group quantities:\n", dp_group_gpu_assignment_with_bandwidth(assignment, all_gpus))
    non_linear_optimize(min_memory_usage, cluster, max_dp_num)
