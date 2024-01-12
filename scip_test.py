from pyomo.environ import *

# 创建一个模型实例
model = ConcreteModel()

# 定义变量
model.x = Var(within=Binary)
model.y = Var(within=NonNegativeIntegers)

# 定义目标函数
model.obj = Objective(expr=2*model.x + 3*model.y, sense=maximize)

# 定义约束
model.c1 = Constraint(expr=model.x + 2*model.y <= 6)

# 创建一个求解器实例
solver = SolverFactory('scip')

# 求解模型
result = solver.solve(model, tee=True)

# 打印结果
print('Status:', result.solver.status)
print('Termination Condition:', result.solver.termination_condition)
print('Objective Value:', model.obj())
print('x =', model.x.value)
print('y =', model.y.value)