import pynvml
import time

pynvml.nvmlInit()
mb = lambda x: x/(1024**2)
device = pynvml.nvmlDeviceGetHandleByIndex(0)

while True:
    x = pynvml.nvmlDeviceGetMemoryInfo(device)
    print("Used: {}MB, total: {}MB".format(mb(x.used), mb(x.total)))
    time.sleep(0.01)