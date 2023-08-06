# tropic_layer

## RoadMap

- [x] optimized operations [MIN|MAX]mm
- [ ] add torch

## Usage

### Get the code

```bash
git clone https://github.com/light5551/tropic_layer.git
cd tropic_layer
```

### Build

#### Using cmake

```bash
mkdir build
cd build
cmake ..
make
./tropics
```

#### Pure

```bash
cd src 
nvcc --std=c++17 main.cu -o main
```

## Profiling

```bash
cd src
nvcc main.cu -o main
nvprof ./main  
```

Result:
```bash
PASSED
==23291== Profiling application: ./tropic_layer
==23291== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   48.91%  2.8800us         1  2.8800us  2.8800us  2.8800us  void maxPlusMulKernel<float>(float const *, float const *, float*, unsigned long, unsigned long, unsigned long)
                   34.24%  2.0160us         3     672ns     544ns     896ns  [CUDA memcpy HtoD]
                   16.85%     992ns         1     992ns     992ns     992ns  [CUDA memcpy DtoH]
      API calls:   99.72%  92.288ms         3  30.763ms  2.4910us  92.282ms  cudaMalloc
                    0.12%  108.43us       101  1.0730us     116ns  45.527us  cuDeviceGetAttribute
                    0.09%  81.928us         3  27.309us  2.7810us  72.809us  cudaFree
                    0.03%  32.082us         4  8.0200us  3.5610us  13.721us  cudaMemcpy
                    0.02%  16.051us         1  16.051us  16.051us  16.051us  cudaLaunchKernel
                    0.01%  10.822us         1  10.822us  10.822us  10.822us  cuDeviceGetName
                    0.01%  8.5710us         1  8.5710us  8.5710us  8.5710us  cuDeviceGetPCIBusId
                    0.00%  1.0490us         3     349ns     163ns     720ns  cuDeviceGetCount
                    0.00%     772ns         1     772ns     772ns     772ns  cuDeviceTotalMem
                    0.00%     655ns         2     327ns     136ns     519ns  cuDeviceGet
                    0.00%     329ns         1     329ns     329ns     329ns  cuDeviceGetUuid

```
