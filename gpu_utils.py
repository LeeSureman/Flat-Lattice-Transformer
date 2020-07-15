from subprocess import Popen, PIPE
from distutils import spawn
import os
import math
import random
import time
import sys
import platform

__version__ = '1.4.0'


class GPU:
    def __init__(self, ID, uuid, load, memoryTotal, memoryUsed, memoryFree, driver, gpu_name, serial, display_mode,
                 display_active, temp_gpu):
        self.id = ID
        self.uuid = uuid
        self.load = load
        self.memoryUtil = float(memoryUsed) / float(memoryTotal)
        self.memoryTotal = memoryTotal
        self.memoryUsed = memoryUsed
        self.memoryFree = memoryFree
        self.driver = driver
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active
        self.temperature = temp_gpu


def safeFloatCast(strNumber):
    try:
        number = float(strNumber)
    except ValueError:
        number = float('nan')
    return number


def getGPUs():
    if platform.system() == "Windows":
        # If the platform is Windows and nvidia-smi
        # could not be found from the environment path,
        # try to find it from system drive with default installation path
        nvidia_smi = spawn.find_executable('nvidia-smi')
        if nvidia_smi is None:
            nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ['systemdrive']
    else:
        nvidia_smi = "nvidia-smi"

    # Get ID, processing and memory utilization for all GPUs
    try:
        p = Popen([nvidia_smi,
                   "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu",
                   "--format=csv,noheader,nounits"], stdout=PIPE)
        stdout, stderror = p.communicate()
    except:
        return []
    output = stdout.decode('UTF-8')
    # output = output[2:-1] # Remove b' and ' from string added by python
    # print(output)
    ## Parse output
    # Split on line break
    lines = output.split(os.linesep)
    # print(lines)
    numDevices = len(lines) - 1
    GPUs = []
    for g in range(numDevices):
        line = lines[g]
        # print(line)
        vals = line.split(', ')
        # print(vals)
        for i in range(12):
            # print(vals[i])
            if (i == 0):
                deviceIds = int(vals[i])
            elif (i == 1):
                uuid = vals[i]
            elif (i == 2):
                gpuUtil = safeFloatCast(vals[i]) / 100
            elif (i == 3):
                memTotal = safeFloatCast(vals[i])
            elif (i == 4):
                memUsed = safeFloatCast(vals[i])
            elif (i == 5):
                memFree = safeFloatCast(vals[i])
            elif (i == 6):
                driver = vals[i]
            elif (i == 7):
                gpu_name = vals[i]
            elif (i == 8):
                serial = vals[i]
            elif (i == 9):
                display_active = vals[i]
            elif (i == 10):
                display_mode = vals[i]
            elif (i == 11):
                temp_gpu = safeFloatCast(vals[i]);
        GPUs.append(GPU(deviceIds, uuid, gpuUtil, memTotal, memUsed, memFree, driver, gpu_name, serial, display_mode,
                        display_active, temp_gpu))
    return GPUs  # (deviceIds, gpuUtil, memUtil)

def getAvailabilityGPU(GPUs, maxLoad=0.5, maxMemory=0.5, memoryFree=0, includeNan=False, excludeID=[], excludeUUID=[]):
    # Determine, which GPUs are available
    GPUavailability = [1 if (gpu.memoryFree>=memoryFree) and (gpu.load < maxLoad or (includeNan and math.isnan(gpu.load))) and (gpu.memoryUtil < maxMemory  or (includeNan and math.isnan(gpu.memoryUtil))) and ((gpu.id not in excludeID) and (gpu.uuid not in excludeUUID)) else 0 for gpu in GPUs]
    gpus = [gpu for gpu, use in zip(GPUs, GPUavailability) if use]
    return gpus


if __name__ == '__main__':
    gpus = getGPUs()
    print(gpus)