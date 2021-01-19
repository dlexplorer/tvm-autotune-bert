import numpy as np
import os
from numpy.lib import ufunclike
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_runtime
import onnx

from tvm.contrib.utils import tempdir
import tvm.contrib.graph_runtime as runtime


def run_tuning(tasks, task_weights, log_file):
    print("Begin tuning...")
    measure_runner = auto_scheduler.RPCRunner(
        "localcpu",
        "127.0.0.1",
        9190,
        min_repeat_ms=300,
        timeout=30,
        repeat=2
    )
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    builder=auto_scheduler.LocalBuilder(build_func="default", n_parallel=1, timeout=10000)
    tune_option = auto_scheduler.TuningOptions(
        builder = builder,
        num_measure_trials=24,
        num_measures_per_round = 2, #just for debugging of jan autotuning issue
        #check_correctness=False,
        #builder_n_parallel=1,
        runner=measure_runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2
    )
    tuner.tune(tune_option)


if __name__ == "__main__":
    #name = "huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad"
    name = "bertsquad10.onnx"
    # target
    target = "llvm -mcpu=core-avx2"
    target_host = "llvm -mcpu=core-avx2"
    # logfile
    log_file = "{name}_{target}".format(
        name=name.replace("/", "_"),
        target="local_cpu"
    )
    print("Extract tasks for " + name + "...")
    onnx_model = onnx.load(name)
    shape_dict = {}
    shape_dict["input_mask:0"] = [1,256]
    shape_dict["segment_ids:0"] = [1,256]
    shape_dict["input_ids:0"] = [1,256]
    shape_dict["unique_ids_raw_output___9:0"] = [1,256]
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    print(log_file)
    if not os.path.exists(log_file):
        tasks, task_weights = auto_scheduler.extract_tasks(
            mod["main"], params, target=target, target_host=target_host)
        for idx, task in enumerate(tasks):
            print("========== Task %d  (workload key: %s) ==========" %
                  (idx, task.workload_key))
            print(task.compute_dag)

        run_tuning(tasks, task_weights, log_file)

