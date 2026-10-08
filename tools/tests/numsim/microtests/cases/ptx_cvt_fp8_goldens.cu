// Ground-truth generator for `ptx_cvt_fp8_goldens.py`.
//
// Every kernel below is one inline-PTX `cvt` spelled exactly as the TIRx
// frontend spells it (`tvm/backend/cuda/intrinsics/cvt.py`), so the goldens pin
// the instruction rather than a CUDA intrinsic wrapper.
//
// Build and run:
//
//     nvcc -gencode=arch=compute_100a,code=sm_100a -o gen ptx_cvt_fp8_goldens.cu
//     CUDA_VISIBLE_DEVICES=<idle gpu> ./gen inputs.bin outputs.bin
//
// `-gencode` is spelled out on purpose: plain `-arch=sm_100a` emits device PTX
// at `.target sm_100`, where ptxas rejects every `.bf16x2` source and
// destination form (PTX ISA 9.1/9.2, `sm_100f` or higher in the same family).
//
// `inputs.bin` holds four sections in the order f32, f16x2, bf16x2, f8x2; each
// section is a little-endian `uint32` count followed by that many little-endian
// `uint32` words (16-bit sources are zero-extended).  Write it from the
// checked-in arrays:
//
//     import struct, numpy as np
//     from tests.numsim.microtests.cases import ptx_cvt_fp8_goldens as g
//     with open("inputs.bin", "wb") as handle:
//         for values in (g.F32_SOURCE, g.F16X2_SOURCE, g.BF16X2_SOURCE, g.F8X2_SOURCE):
//             handle.write(struct.pack("<I", len(values)))
//             handle.write(values.astype("<u4").tobytes())
//
// `outputs.bin` uses the same section format, one section per row of
// `FORM_ORDER` below, which is the key order of `GOLDENS`.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CHECK(call)                                                                \
  do {                                                                             \
    cudaError_t status = (call);                                                   \
    if (status != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at line %d\n", cudaGetErrorString(status),     \
              __LINE__);                                                           \
      exit(1);                                                                     \
    }                                                                              \
  } while (0)

// cvt.rn.satfinite{.relu}.f8x2type.f32   d, a, b;
// `a` is the upper byte of `d`; the pairing below places every source value in
// both operand positions.
#define F8_FROM_F32(NAME, INSN)                                                    \
  __global__ void NAME(unsigned int* out, const unsigned int* in, int n) {         \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (i >= n) return;                                                            \
    float a = __int_as_float(in[i]);                                               \
    float b = __int_as_float(in[(i + 1) % n]);                                     \
    unsigned short d;                                                              \
    asm volatile(INSN " %0, %1, %2;" : "=h"(d) : "f"(a), "f"(b));                  \
    out[i] = d;                                                                    \
  }

F8_FROM_F32(a_e4m3, "cvt.rn.satfinite.e4m3x2.f32")
F8_FROM_F32(a_e4m3_relu, "cvt.rn.satfinite.relu.e4m3x2.f32")
F8_FROM_F32(a_e5m2, "cvt.rn.satfinite.e5m2x2.f32")
F8_FROM_F32(a_e5m2_relu, "cvt.rn.satfinite.relu.e5m2x2.f32")

// cvt.rn.satfinite{.relu}.f8x2type.fp16x2   d, a;
#define F8_FROM_PACKED(NAME, INSN)                                                 \
  __global__ void NAME(unsigned int* out, const unsigned int* in, int n) {         \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (i >= n) return;                                                            \
    unsigned int a = in[i];                                                        \
    unsigned short d;                                                              \
    asm volatile(INSN " %0, %1;" : "=h"(d) : "r"(a));                              \
    out[i] = d;                                                                    \
  }

F8_FROM_PACKED(b_e4m3, "cvt.rn.satfinite.e4m3x2.f16x2")
F8_FROM_PACKED(b_e4m3_relu, "cvt.rn.satfinite.relu.e4m3x2.f16x2")
F8_FROM_PACKED(b_e5m2, "cvt.rn.satfinite.e5m2x2.f16x2")
F8_FROM_PACKED(b_e5m2_relu, "cvt.rn.satfinite.relu.e5m2x2.f16x2")
F8_FROM_PACKED(c_e4m3, "cvt.rn.satfinite.e4m3x2.bf16x2")
F8_FROM_PACKED(c_e4m3_relu, "cvt.rn.satfinite.relu.e4m3x2.bf16x2")
F8_FROM_PACKED(c_e5m2, "cvt.rn.satfinite.e5m2x2.bf16x2")
F8_FROM_PACKED(c_e5m2_relu, "cvt.rn.satfinite.relu.e5m2x2.bf16x2")

// cvt.rn{.relu}.f16x2.f8x2type   d, a;  and
// cvt.rn{.relu}{.satfinite}.bf16x2.f8x2type   d, a;
#define PACKED_FROM_F8(NAME, INSN)                                                 \
  __global__ void NAME(unsigned int* out, const unsigned int* in, int n) {         \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (i >= n) return;                                                            \
    unsigned short a = (unsigned short)in[i];                                      \
    unsigned int d;                                                                \
    asm volatile(INSN " %0, %1;" : "=r"(d) : "h"(a));                              \
    out[i] = d;                                                                    \
  }

PACKED_FROM_F8(d_e4m3, "cvt.rn.f16x2.e4m3x2")
PACKED_FROM_F8(d_e4m3_relu, "cvt.rn.relu.f16x2.e4m3x2")
PACKED_FROM_F8(d_e5m2, "cvt.rn.f16x2.e5m2x2")
PACKED_FROM_F8(d_e5m2_relu, "cvt.rn.relu.f16x2.e5m2x2")
PACKED_FROM_F8(e_e4m3, "cvt.rn.bf16x2.e4m3x2")
PACKED_FROM_F8(e_e4m3_relu, "cvt.rn.relu.bf16x2.e4m3x2")
PACKED_FROM_F8(e_e4m3_sat, "cvt.rn.satfinite.bf16x2.e4m3x2")
PACKED_FROM_F8(e_e4m3_relu_sat, "cvt.rn.relu.satfinite.bf16x2.e4m3x2")
PACKED_FROM_F8(e_e5m2, "cvt.rn.bf16x2.e5m2x2")
PACKED_FROM_F8(e_e5m2_relu, "cvt.rn.relu.bf16x2.e5m2x2")
PACKED_FROM_F8(e_e5m2_sat, "cvt.rn.satfinite.bf16x2.e5m2x2")
PACKED_FROM_F8(e_e5m2_relu_sat, "cvt.rn.relu.satfinite.bf16x2.e5m2x2")

typedef void (*Kernel)(unsigned int*, const unsigned int*, int);

struct Form {
  const char* name;
  Kernel kernel;
  int section;  // 0 = f32, 1 = f16x2, 2 = bf16x2, 3 = f8x2
};

static const Form FORM_ORDER[] = {
    {"e4m3x2.f32", a_e4m3, 0},
    {"e4m3x2.f32.relu", a_e4m3_relu, 0},
    {"e5m2x2.f32", a_e5m2, 0},
    {"e5m2x2.f32.relu", a_e5m2_relu, 0},
    {"e4m3x2.f16x2", b_e4m3, 1},
    {"e4m3x2.f16x2.relu", b_e4m3_relu, 1},
    {"e5m2x2.f16x2", b_e5m2, 1},
    {"e5m2x2.f16x2.relu", b_e5m2_relu, 1},
    {"e4m3x2.bf16x2", c_e4m3, 2},
    {"e4m3x2.bf16x2.relu", c_e4m3_relu, 2},
    {"e5m2x2.bf16x2", c_e5m2, 2},
    {"e5m2x2.bf16x2.relu", c_e5m2_relu, 2},
    {"f16x2.e4m3x2", d_e4m3, 3},
    {"f16x2.e4m3x2.relu", d_e4m3_relu, 3},
    {"f16x2.e5m2x2", d_e5m2, 3},
    {"f16x2.e5m2x2.relu", d_e5m2_relu, 3},
    {"bf16x2.e4m3x2", e_e4m3, 3},
    {"bf16x2.e4m3x2.relu", e_e4m3_relu, 3},
    {"bf16x2.e4m3x2.satfinite", e_e4m3_sat, 3},
    {"bf16x2.e4m3x2.relu.satfinite", e_e4m3_relu_sat, 3},
    {"bf16x2.e5m2x2", e_e5m2, 3},
    {"bf16x2.e5m2x2.relu", e_e5m2_relu, 3},
    {"bf16x2.e5m2x2.satfinite", e_e5m2_sat, 3},
    {"bf16x2.e5m2x2.relu.satfinite", e_e5m2_relu_sat, 3},
};

static std::vector<unsigned int> read_section(FILE* file) {
  unsigned int count = 0;
  if (fread(&count, sizeof(count), 1, file) != 1) {
    fprintf(stderr, "truncated input file\n");
    exit(1);
  }
  std::vector<unsigned int> values(count);
  if (count && fread(values.data(), sizeof(unsigned int), count, file) != count) {
    fprintf(stderr, "truncated input section\n");
    exit(1);
  }
  return values;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <inputs.bin> <outputs.bin>\n", argv[0]);
    return 1;
  }
  FILE* input_file = fopen(argv[1], "rb");
  if (!input_file) {
    fprintf(stderr, "cannot open %s\n", argv[1]);
    return 1;
  }
  std::vector<unsigned int> sections[4];
  for (int index = 0; index < 4; ++index) sections[index] = read_section(input_file);
  fclose(input_file);

  cudaDeviceProp properties;
  CHECK(cudaGetDeviceProperties(&properties, 0));
  int driver_version = 0;
  CHECK(cudaDriverGetVersion(&driver_version));
  fprintf(stderr, "device=%s compute=%d.%d driver=%d\n", properties.name, properties.major,
          properties.minor, driver_version);

  size_t widest = 0;
  for (int index = 0; index < 4; ++index)
    if (sections[index].size() > widest) widest = sections[index].size();
  unsigned int* device_input = nullptr;
  unsigned int* device_output = nullptr;
  CHECK(cudaMalloc(&device_input, widest * sizeof(unsigned int)));
  CHECK(cudaMalloc(&device_output, widest * sizeof(unsigned int)));

  FILE* output_file = fopen(argv[2], "wb");
  if (!output_file) {
    fprintf(stderr, "cannot open %s\n", argv[2]);
    return 1;
  }

  const int form_count = sizeof(FORM_ORDER) / sizeof(FORM_ORDER[0]);
  for (int index = 0; index < form_count; ++index) {
    const Form& form = FORM_ORDER[index];
    const std::vector<unsigned int>& values = sections[form.section];
    int count = (int)values.size();
    CHECK(cudaMemcpy(device_input, values.data(), count * sizeof(unsigned int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemset(device_output, 0, count * sizeof(unsigned int)));
    int threads = 128;
    int blocks = (count + threads - 1) / threads;
    form.kernel<<<blocks, threads>>>(device_output, device_input, count);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<unsigned int> results(count);
    CHECK(cudaMemcpy(results.data(), device_output, count * sizeof(unsigned int),
                     cudaMemcpyDeviceToHost));
    unsigned int written = (unsigned int)count;
    fwrite(&written, sizeof(written), 1, output_file);
    fwrite(results.data(), sizeof(unsigned int), count, output_file);
    fprintf(stderr, "form %-32s n=%d\n", form.name, count);
  }
  fclose(output_file);
  return 0;
}
