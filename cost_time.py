import torch
import numpy as np
import torchvision.models as models
import time

model = models.densenet121(pretrained=True)

t0 = time.time()
model = model.cuda()

dummy_input = torch.randn(10, 3, 224, 224, dtype=torch.float).cuda()
dummy_input2 = torch.randn(10, 3, 224, 224, dtype=torch.float).cuda()
dummy_input3 = torch.randn(10, 3, 224, 224, dtype=torch.float).cuda()
dummy_input4 = torch.randn(10, 3, 224, 224, dtype=torch.float).cuda()
dummy_input5 = torch.randn(10, 3, 224, 224, dtype=torch.float).cuda()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 100
timings = np.zeros((repetitions, 1))
# GPU-WARM-UP
for _ in range(10):
    _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        _ = model(dummy_input5)
        _ = model(dummy_input2)
        _ = model(dummy_input3)
        _ = model(dummy_input4)


        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)

        timings[rep] = curr_time
t1 = time.time()
print(t1-t0)
print(np.sum(timings))
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn)
