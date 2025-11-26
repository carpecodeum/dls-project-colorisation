"""Logic for backend selection"""
import os
from needle import backend_ndarray as nd_backend


BACKEND = os.environ.get("NEEDLE_BACKEND", "nd")

if BACKEND == "nd":
    from needle import backend_ndarray as array_api
    from needle.backend_ndarray import (
        all_devices,
        cuda,
        cpu,
        cpu_numpy,
        default_device,
        BackendDevice as Device,
    )

    NDArray = array_api.NDArray
else:
    import numpy as array_api
    from needle import backend_numpy as device_module

    all_devices = device_module.all_devices
    cuda = device_module.cuda
    cpu = device_module.cpu
    default_device = device_module.default_device
    Device = device_module.Device

    NDArray = array_api.ndarray


def cuda():
    """Return cuda device"""
    try:
        from needle.backend_ndarray import cuda

        return cuda()
    except:
        return None
