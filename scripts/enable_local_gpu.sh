#!/bin/bash
export PYTENSOR_FLAGS="mode=JAX"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.75"
echo "JAX and PyTensor GPU limits activated for local run."