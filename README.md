# To reproduce a fail of compilation with tune information
1. Download bertsquad10.onnx from ONNX model zoo: https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad/model
2. Downnlaad TVM, build with llvm support, set up environment variables to find tvm python module 
3. In separate console run rpc tracker: python -m tvm.exec.rpc_tracker --host 127.0.0.1 --port 9190
4. In separate console run rpc server: python -m tvm.exec.rpc_server --tracker 127.0.0.1:9190 --port 9090 --key localcpu --no-fork
5. Execute search_bert_local.py: python search_bert_local.py
6. After finishing of statistic collection in intem 5, execute compile_bert_tuned_local.py: python compile_bert_tuned_local.py

There will be a fail:
  Check failed: false == false: [22:31:18] src/relay/op/nn/nn.h:73: 
---------------------------------------------------------------
An internal invariant was violated during the execution of TVM.
Please read TVM's error reporting guidelines.
More details can be found here: https://discuss.tvm.ai/t/error-reporting/7793.
---------------------------------------------------------------
  Check failed: static_cast<int>(weight->shape.size()) == 2 == false: 

 
- 
