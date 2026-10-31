"""NVML measures CUDA and Vulkan allocations together, including driver memory."""

import ctypes
import os
import resource
import threading


class ProcessMemory(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used", ctypes.c_ulonglong),
        ("gpu", ctypes.c_uint),
        ("compute", ctypes.c_uint),
    ]


class Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class ResourceMonitor:
    def __init__(self):
        self.nvml = ctypes.CDLL("/run/opengl-driver/lib/libnvidia-ml.so.1")
        if self.nvml.nvmlInit_v2() != 0:
            raise RuntimeError("NVML initialization failed")
        self.handle = ctypes.c_void_p()
        if self.nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
            raise RuntimeError("NVML device unavailable")
        self.peak_vram = 0
        self.utilization = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        while not self.stop.is_set():
            for name in (
                "nvmlDeviceGetComputeRunningProcesses_v2",
                "nvmlDeviceGetGraphicsRunningProcesses_v2",
            ):
                count = ctypes.c_uint(64)
                entries = (ProcessMemory * 64)()
                if (
                    getattr(self.nvml, name)(self.handle, ctypes.byref(count), entries)
                    == 0
                ):
                    for item in entries[: count.value]:
                        if item.pid == os.getpid() and item.used < 2**63:
                            self.peak_vram = max(self.peak_vram, item.used)
            util = Utilization()
            if (
                self.nvml.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util))
                == 0
            ):
                self.utilization.append(util.gpu)
            self.stop.wait(0.02)

    def finish(self):
        self.stop.set()
        self.thread.join()
        return {
            "peak_process_vram_mib": self.peak_vram / 1048576,
            "peak_process_ram_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "gpu_busy_max_percent": max(self.utilization, default=0),
        }
