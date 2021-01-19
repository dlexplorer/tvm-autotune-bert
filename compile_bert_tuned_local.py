import argparse
import numpy as np
import os
from numpy.lib import ufunclike
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_runtime
import onnx
from tvm.contrib import ndk

from tvm.contrib.utils import tempdir
import tvm.contrib.graph_runtime as runtime

name = "bertsquad10.onnx"
# target
target = "llvm -mcpu=core-avx2"
target_host = "llvm -mcpu=core-avx2"
# logfile
log_file = "{name}_{target}".format(
    name=name.replace("/", "_"),
    target="local_cpu"
)

onnx_model = onnx.load(name)
shape_dict = {}
shape_dict["input_mask:0"] = [1,256]
shape_dict["segment_ids:0"] = [1,256]
shape_dict["input_ids:0"] = [1,256]
shape_dict["unique_ids_raw_output___9:0"] = [1,256]
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target,
                            target_host=target_host, params=params)

lib.export_library(name + "arm.tune.so", ndk.create_shared)
