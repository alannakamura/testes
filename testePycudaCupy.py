# -*- coding: utf-8 -*-
import numpy as np
import cupy as cp

# ---------- 1) Inicializa CuPy e PRIMARY CONTEXT ----------
cp.cuda.Device(0).use()
_ = cp.zeros(1, dtype=cp.float32)  # força criação do primary context

# ---------- 2) Anexa PyCUDA ao MESMO primary context ----------
import pycuda.driver as drv
drv.init()
dev = drv.Device(0)
ctx = dev.retain_primary_context()  # pega o primary context (mesmo do CuPy)
ctx.push()                          # torna-o corrente para PyCUDA

from pycuda.compiler import SourceModule

# ---------- 3) Compila kernel DEPOIS do contexto estar fixo ----------
mod = SourceModule(r"""
extern "C" __global__
void vec_add(const float *a, const float *b, float *c, int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}
""")
vec_add = mod.get_function("vec_add")

# ---------- 4) Aloca buffers com PyCUDA ----------
n = 5
a_h = np.arange(n, dtype=np.float32)
b_h = np.arange(1, 1+n, dtype=np.float32)
c_h = np.empty_like(a_h)

a_d = drv.mem_alloc(a_h.nbytes)
b_d = drv.mem_alloc(b_h.nbytes)
c_d = drv.mem_alloc(c_h.nbytes)

drv.memcpy_htod(a_d, a_h)
drv.memcpy_htod(b_d, b_h)

# ---------- 5) Cria views CuPy SOBRE os ponteiros do PyCUDA ----------
def as_cupy_view(dev_alloc, shape, dtype):
    ptr = int(dev_alloc)  # CUdeviceptr -> inteiro
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    # owner=None: CuPy NÃO tentará liberar (PyCUDA continua dono)
    mem = cp.cuda.UnownedMemory(ptr, nbytes, owner=None)
    mp = cp.cuda.MemoryPointer(mem, 0)
    return cp.ndarray(shape, dtype=dtype, memptr=mp)

a_cp = as_cupy_view(a_d, a_h.shape, a_h.dtype)
b_cp = as_cupy_view(b_d, b_h.shape, b_h.dtype)

# ---------- 6) Opera com CuPy na GPU ----------
a_cp += 1
b_cp += 1
cp.cuda.Stream.null.synchronize()  # garante término antes do kernel

# ---------- 7) Lança kernel PyCUDA no MESMO contexto ----------
block = 128
grid = (n + block - 1) // block
vec_add(a_d, b_d, c_d, np.int32(n),
        block=(block, 1, 1), grid=(grid, 1, 1))

drv.memcpy_dtoh(c_h, c_d)

print("a host:", a_h)        # [0 1 2 ...]
print("b host:", b_h)        # [0 1 2 ...]
print("c (= (a+1)+(b+1)):", c_h)

# ---------- 8) Limpeza (opcional) ----------
a_d.free(); b_d.free(); c_d.free()
ctx.pop()   # se você gerencia contexto manualmente
