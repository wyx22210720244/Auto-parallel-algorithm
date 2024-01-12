class NODE:
    def __init__(self,node_id):
        self.node_id = node_id
        self.gpus = []

    def add_gpu(self,gpu):
        self.gpus.append(gpu)

    def __str__(self):
        gpu_str = ",".join(f"{gpu})" for idx,gpu in enumerate(self.gpus))
        return f"(node_id={self.node_id},GPU={gpu_str})"

class Cluster:
    def __init__(self):
        self.nodes = []
        self.num_nodes = 0
        self.num_gpus = 0

    def add_node(self,node):
        self.nodes.append(node)

    def __str__(self):
        return f"Cluster(Nodes={[str(node) for node in self.nodes]})"
class GPU:
    def __init__(self,name,compute_power,memory,local_rank=None,node=None):
        self.name = name
        self.compute_power = compute_power
        self.memory = memory
        self.local_rank = local_rank
        self.node = node

    def __str__(self):
        return f"GPU(name={self.name},compute_power={self.compute_power},memory={self.memory},local_rank={self.local_rank},node={self.node})"

    def __repr__(self):
        return str(self)

    def to_list(self):
        return [self.name,self.compute_power,self.memory]
