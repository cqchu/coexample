from megengine.utils import custom_op_tools
from megengine.core._imperative_rt.core2 import apply
from megengine.core.ops import custom
from megengine.tensor import Parameter, Tensor

import os
import numpy as np

lib_path = custom_op_tools.build_and_load(
    # op的名字
    "matmul_scale",
    # custom op相关的代码路径
    sources=["custom_opsrc/matmul_scale.cpp", "custom_opsrc/matmul_scale.cu"],
    # 一些c的编译flag
    extra_cflags=[],
    # 一些cuda编译flag，比如指定gpu架构信息等
    extra_cuda_cflags=[],
    # 指定一些链接信息，比如想链接一些其他计算库，如果使用cudnn，可以使用下面这个 ldflags
    extra_ldflags=[],              
    # extra_ldflags=["-lcudnn -L~/.local/cuda-10.2-cudnn-7.6.5-tensorrt-6.0.1.8/cudnn/lib64 -Wl,-rpath,~/.local/cuda-10.2-cudnn-7.6.5-tensorrt-6.0.1.8/cudnn/lib64"], 
    # 指定一些include path
    extra_include_paths=[
        "./custom_opsrc",
        "~/.local/cuda-10.2-cudnn-7.6.5-tensorrt-6.0.1.8/cuda/include"],
    # 指定你想把编译出来的.so放在什么目录
    build_dir="./build",
    # 是否展示一些中间编译过程
    verbose=False,
)

op = custom.MatMulScaleForward(scale = 0.1)
lhs = Tensor(np.random.uniform(size=(2, 4)).astype("float32"))
rhs = Tensor(np.random.uniform(size=(4, 3)).astype("float32"))
result = apply(op, lhs, rhs)
print(result)

os.system("rm -r build")
