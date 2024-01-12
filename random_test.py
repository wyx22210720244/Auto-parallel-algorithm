import torch
import torch.distributed as dist


def get_gpu_type(rank):
    gpu_type = torch.cuda.get_device_properties(rank).name
    return gpu_type


dist.init_process_group(backend='gloo')
rank = dist.get_rank()
world_size = dist.get_world_size()
print(f"world_size: {world_size}")
# gpu_type = get_gpu_type(rank)
# print(f"rank: {rank}, gpu_name: {gpu_type}")
# device = torch.device(f'cuda:{rank}')
# torch.cuda.set_device(device)
# gpu_fp16 = gpu_database.get_gpu_infor('gpu_info',gpu_type,'gpu_fp16')[0][0]
input_tensor = torch.tensor(torch.ones(1), dtype=torch.float32)
all_tyles = [torch.zeros(1, dtype=torch.float32) for i in range(world_size)]
dist.gather(input_tensor, all_tyles if rank == 0 else None, 0)
print(f"rank: {rank}, all_tyles: {all_tyles}")
