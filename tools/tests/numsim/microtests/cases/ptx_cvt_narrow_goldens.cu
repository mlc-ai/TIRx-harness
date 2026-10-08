// Ground-truth generator for `ptx_cvt_narrow_goldens.py`.
//
// Companion to `ptx_cvt_fp8_goldens.cu`, covering the rest of the packed `cvt`
// family: the `.e2m1x2` pack/unpack forms, the `.rs` stochastic four-packs, the
// `.ue8m0x2` roundings, and the `.scaled::n2::ue8m0` operand forms.  Every
// kernel is one inline-PTX `cvt` spelled exactly as the TIRx frontend spells it
// (`tvm/backend/cuda/intrinsics/cvt.py`), including the `.b8` staging that
// frontend wraps around every `e2m1x2` operand and result, so the goldens pin
// the instruction as a kernel can actually reach it.
//
// Build and run:
//
//     nvcc -gencode=arch=compute_100a,code=sm_100a -o gen ptx_cvt_narrow_goldens.cu
//     CUDA_VISIBLE_DEVICES=<idle gpu> ./gen inputs.bin outputs.bin
//
// `-gencode` is spelled out on purpose: plain `-arch=sm_100a` emits device PTX
// at `.target sm_100`, where ptxas rejects every spelling below.
//
// `inputs.bin` holds eight sections in the order f32, f16x2, bf16x2, f4x2, f8x2,
// ue8m0x2, scale, rbits; each section is a little-endian `uint32` count followed
// by that many little-endian `uint32` words (narrower sources are zero-extended).
// Write it from the checked-in arrays:
//
//     import struct
//     from tests.numsim.microtests.cases import ptx_cvt_narrow_goldens as g
//     with open("inputs.bin", "wb") as handle:
//         for values in g.SECTIONS:
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

// cvt.rn.satfinite{.relu}.e2m1x2.f32   d, a, b;   (d staged through a .b8)
// `a` is the upper nibble of `d`; the pairing places every source value in both
// operand positions.
#define F4_FROM_F32(NAME, INSN)                                                    \
  __global__ void NAME(unsigned int* out, const unsigned int* a,                   \
                       const unsigned int* b, int n) {                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (i >= n) return;                                                            \
    (void)b;                                                                       \
    float x = __int_as_float(a[i]);                                                \
    float y = __int_as_float(a[(i + 1) % n]);                                      \
    unsigned short d;                                                              \
    asm volatile("{\n\t.reg .b8 raw_result;\n\t" INSN                              \
                 " raw_result, %1, %2;\n\tcvt.u16.u8 %0, raw_result;\n\t}"         \
                 : "=h"(d)                                                         \
                 : "f"(x), "f"(y));                                                \
    out[i] = d;                                                                    \
  }

// cvt.rn.satfinite{.relu}.e2m1x2.fp16x2   d, a;
#define F4_FROM_PACKED(NAME, INSN)                                                 \
  __global__ void NAME(unsigned int* out, const unsigned int* a,                   \
                       const unsigned int* b, int n) {                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (i >= n) return;                                                            \
    (void)b;                                                                       \
    unsigned int x = a[i];                                                         \
    unsigned short d;                                                              \
    asm volatile("{\n\t.reg .b8 raw_result;\n\t" INSN                              \
                 " raw_result, %1;\n\tcvt.u16.u8 %0, raw_result;\n\t}"             \
                 : "=h"(d)                                                         \
                 : "r"(x));                                                        \
    out[i] = d;                                                                    \
  }

// cvt.rn{.relu}.f16x2.e2m1x2   d, a;   and
// cvt.rn{.relu}{.satfinite}.bf16x2.e2m1x2   d, a;   (a staged through a .b8)
#define PACKED_FROM_F4(NAME, INSN)                                                 \
  __global__ void NAME(unsigned int* out, const unsigned int* a,                   \
                       const unsigned int* b, int n) {                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (i >= n) return;                                                            \
    (void)b;                                                                       \
    unsigned short x = (unsigned short)a[i];                                       \
    unsigned int d;                                                                \
    asm volatile("{\n\t.reg .b8 raw_a;\n\tcvt.u8.u16 raw_a, %1;\n\t" INSN          \
                 " %0, raw_a;\n\t}"                                                \
                 : "=r"(d)                                                         \
                 : "h"(x));                                                        \
    out[i] = d;                                                                    \
  }

// cvt.rs{.relu}.satfinite.f4x4type.f32   d, {a, b, e, f}, rbits;
#define RS4_U16(NAME, INSN)                                                        \
  __global__ void NAME(unsigned int* out, const unsigned int* a,                   \
                       const unsigned int* b, int n) {                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (i >= n) return;                                                            \
    float p = __int_as_float(a[i]);                                                \
    float q = __int_as_float(a[(i + 1) % n]);                                      \
    float r = __int_as_float(a[(i + 2) % n]);                                      \
    float s = __int_as_float(a[(i + 3) % n]);                                      \
    unsigned short d;                                                              \
    asm volatile(INSN " %0, {%1, %2, %3, %4}, %5;"                                 \
                 : "=h"(d)                                                         \
                 : "f"(p), "f"(q), "f"(r), "f"(s), "r"(b[i]));                     \
    out[i] = d;                                                                    \
  }

// cvt.rs{.relu}.satfinite.f8x4type.f32   d, {a, b, e, f}, rbits;
#define RS4_U32(NAME, INSN)                                                        \
  __global__ void NAME(unsigned int* out, const unsigned int* a,                   \
                       const unsigned int* b, int n) {                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (i >= n) return;                                                            \
    float p = __int_as_float(a[i]);                                                \
    float q = __int_as_float(a[(i + 1) % n]);                                      \
    float r = __int_as_float(a[(i + 2) % n]);                                      \
    float s = __int_as_float(a[(i + 3) % n]);                                      \
    unsigned int d;                                                                \
    asm volatile(INSN " %0, {%1, %2, %3, %4}, %5;"                                 \
                 : "=r"(d)                                                         \
                 : "f"(p), "f"(q), "f"(r), "f"(s), "r"(b[i]));                     \
    out[i] = d;                                                                    \
  }

// cvt.{rz,rp}{.satfinite}.ue8m0x2.f32   d, a, b;
#define UE8_FROM_F32(NAME, INSN)                                                   \
  __global__ void NAME(unsigned int* out, const unsigned int* a,                   \
                       const unsigned int* b, int n) {                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (i >= n) return;                                                            \
    (void)b;                                                                       \
    float x = __int_as_float(a[i]);                                                \
    float y = __int_as_float(a[(i + 1) % n]);                                      \
    unsigned short d;                                                              \
    asm volatile(INSN " %0, %1, %2;" : "=h"(d) : "f"(x), "f"(y));                  \
    out[i] = d;                                                                    \
  }

// cvt.{rz,rp}{.satfinite}.ue8m0x2.bf16x2   d, a;
#define UE8_FROM_PACKED(NAME, INSN)                                                \
  __global__ void NAME(unsigned int* out, const unsigned int* a,                   \
                       const unsigned int* b, int n) {                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (i >= n) return;                                                            \
    (void)b;                                                                       \
    unsigned short d;                                                              \
    asm volatile(INSN " %0, %1;" : "=h"(d) : "r"(a[i]));                           \
    out[i] = d;                                                                    \
  }

// cvt.rn.bf16x2.ue8m0x2   d, a;
#define PACKED_FROM_UE8(NAME, INSN)                                                \
  __global__ void NAME(unsigned int* out, const unsigned int* a,                   \
                       const unsigned int* b, int n) {                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (i >= n) return;                                                            \
    (void)b;                                                                       \
    unsigned short x = (unsigned short)a[i];                                       \
    unsigned int d;                                                                \
    asm volatile(INSN " %0, %1;" : "=r"(d) : "h"(x));                              \
    out[i] = d;                                                                    \
  }

// cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.f8x2type   d, a, scale;
#define SCALED_FROM_F8(NAME, INSN)                                                 \
  __global__ void NAME(unsigned int* out, const unsigned int* a,                   \
                       const unsigned int* b, int n) {                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (i >= n) return;                                                            \
    unsigned short x = (unsigned short)a[i];                                       \
    unsigned short scale = (unsigned short)b[i];                                   \
    unsigned int d;                                                                \
    asm volatile(INSN " %0, %1, %2;" : "=r"(d) : "h"(x), "h"(scale));              \
    out[i] = d;                                                                    \
  }

// cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.e2m1x2   d, a, scale;
#define SCALED_FROM_F4(NAME, INSN)                                                 \
  __global__ void NAME(unsigned int* out, const unsigned int* a,                   \
                       const unsigned int* b, int n) {                             \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
    if (i >= n) return;                                                            \
    unsigned short x = (unsigned short)a[i];                                       \
    unsigned short scale = (unsigned short)b[i];                                   \
    unsigned int d;                                                                \
    asm volatile("{\n\t.reg .b8 raw_a;\n\tcvt.u8.u16 raw_a, %1;\n\t" INSN          \
                 " %0, raw_a, %2;\n\t}"                                            \
                 : "=r"(d)                                                         \
                 : "h"(x), "h"(scale));                                            \
    out[i] = d;                                                                    \
  }

F4_FROM_F32(a_f4, "cvt.rn.satfinite.e2m1x2.f32")
F4_FROM_F32(a_f4_relu, "cvt.rn.satfinite.relu.e2m1x2.f32")
F4_FROM_PACKED(b_f4_f16, "cvt.rn.satfinite.e2m1x2.f16x2")
F4_FROM_PACKED(b_f4_f16_relu, "cvt.rn.satfinite.relu.e2m1x2.f16x2")
F4_FROM_PACKED(c_f4_bf16, "cvt.rn.satfinite.e2m1x2.bf16x2")
F4_FROM_PACKED(c_f4_bf16_relu, "cvt.rn.satfinite.relu.e2m1x2.bf16x2")

PACKED_FROM_F4(d_f16, "cvt.rn.f16x2.e2m1x2")
PACKED_FROM_F4(d_f16_relu, "cvt.rn.relu.f16x2.e2m1x2")
PACKED_FROM_F4(e_bf16, "cvt.rn.bf16x2.e2m1x2")
PACKED_FROM_F4(e_bf16_relu, "cvt.rn.relu.bf16x2.e2m1x2")
PACKED_FROM_F4(e_bf16_sat, "cvt.rn.satfinite.bf16x2.e2m1x2")
PACKED_FROM_F4(e_bf16_relu_sat, "cvt.rn.relu.satfinite.bf16x2.e2m1x2")

RS4_U16(f_rs_f4, "cvt.rs.satfinite.e2m1x4.f32")
RS4_U16(f_rs_f4_relu, "cvt.rs.relu.satfinite.e2m1x4.f32")
RS4_U32(f_rs_e4m3, "cvt.rs.satfinite.e4m3x4.f32")
RS4_U32(f_rs_e4m3_relu, "cvt.rs.relu.satfinite.e4m3x4.f32")
RS4_U32(f_rs_e5m2, "cvt.rs.satfinite.e5m2x4.f32")
RS4_U32(f_rs_e5m2_relu, "cvt.rs.relu.satfinite.e5m2x4.f32")

UE8_FROM_F32(g_ue8_rz, "cvt.rz.ue8m0x2.f32")
UE8_FROM_F32(g_ue8_rz_sat, "cvt.rz.satfinite.ue8m0x2.f32")
UE8_FROM_F32(g_ue8_rp, "cvt.rp.ue8m0x2.f32")
UE8_FROM_F32(g_ue8_rp_sat, "cvt.rp.satfinite.ue8m0x2.f32")
UE8_FROM_PACKED(h_ue8_rz, "cvt.rz.ue8m0x2.bf16x2")
UE8_FROM_PACKED(h_ue8_rz_sat, "cvt.rz.satfinite.ue8m0x2.bf16x2")
UE8_FROM_PACKED(h_ue8_rp, "cvt.rp.ue8m0x2.bf16x2")
UE8_FROM_PACKED(h_ue8_rp_sat, "cvt.rp.satfinite.ue8m0x2.bf16x2")
PACKED_FROM_UE8(i_bf16_ue8, "cvt.rn.bf16x2.ue8m0x2")

SCALED_FROM_F8(j_e4m3, "cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2")
SCALED_FROM_F8(j_e4m3_relu, "cvt.rn.relu.scaled::n2::ue8m0.bf16x2.e4m3x2")
SCALED_FROM_F8(j_e4m3_sat, "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.e4m3x2")
SCALED_FROM_F8(j_e4m3_relu_sat, "cvt.rn.relu.satfinite.scaled::n2::ue8m0.bf16x2.e4m3x2")
SCALED_FROM_F8(k_e5m2, "cvt.rn.scaled::n2::ue8m0.bf16x2.e5m2x2")
SCALED_FROM_F8(k_e5m2_relu, "cvt.rn.relu.scaled::n2::ue8m0.bf16x2.e5m2x2")
SCALED_FROM_F8(k_e5m2_sat, "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.e5m2x2")
SCALED_FROM_F8(k_e5m2_relu_sat, "cvt.rn.relu.satfinite.scaled::n2::ue8m0.bf16x2.e5m2x2")
SCALED_FROM_F4(l_e2m1, "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2")
SCALED_FROM_F4(l_e2m1_relu, "cvt.rn.relu.scaled::n2::ue8m0.bf16x2.e2m1x2")
SCALED_FROM_F4(l_e2m1_sat, "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.e2m1x2")
SCALED_FROM_F4(l_e2m1_relu_sat, "cvt.rn.relu.satfinite.scaled::n2::ue8m0.bf16x2.e2m1x2")

typedef void (*Kernel)(unsigned int*, const unsigned int*, const unsigned int*, int);

enum Section { F32 = 0, F16X2 = 1, BF16X2 = 2, F4X2 = 3, F8X2 = 4, UE8X2 = 5, SCALE = 6, RBITS = 7 };

struct Form {
  const char* name;
  Kernel kernel;
  int primary;    // section feeding the primary operands
  int secondary;  // section feeding rbits or the scale factor, or -1
};

static const Form FORM_ORDER[] = {
    {"e2m1x2.f32", a_f4, F32, -1},
    {"e2m1x2.f32.relu", a_f4_relu, F32, -1},
    {"e2m1x2.f16x2", b_f4_f16, F16X2, -1},
    {"e2m1x2.f16x2.relu", b_f4_f16_relu, F16X2, -1},
    {"e2m1x2.bf16x2", c_f4_bf16, BF16X2, -1},
    {"e2m1x2.bf16x2.relu", c_f4_bf16_relu, BF16X2, -1},
    {"f16x2.e2m1x2", d_f16, F4X2, -1},
    {"f16x2.e2m1x2.relu", d_f16_relu, F4X2, -1},
    {"bf16x2.e2m1x2", e_bf16, F4X2, -1},
    {"bf16x2.e2m1x2.relu", e_bf16_relu, F4X2, -1},
    {"bf16x2.e2m1x2.satfinite", e_bf16_sat, F4X2, -1},
    {"bf16x2.e2m1x2.relu.satfinite", e_bf16_relu_sat, F4X2, -1},
    {"e2m1x4.f32.rs", f_rs_f4, F32, RBITS},
    {"e2m1x4.f32.rs.relu", f_rs_f4_relu, F32, RBITS},
    {"e4m3x4.f32.rs", f_rs_e4m3, F32, RBITS},
    {"e4m3x4.f32.rs.relu", f_rs_e4m3_relu, F32, RBITS},
    {"e5m2x4.f32.rs", f_rs_e5m2, F32, RBITS},
    {"e5m2x4.f32.rs.relu", f_rs_e5m2_relu, F32, RBITS},
    {"ue8m0x2.f32.rz", g_ue8_rz, F32, -1},
    {"ue8m0x2.f32.rz.satfinite", g_ue8_rz_sat, F32, -1},
    {"ue8m0x2.f32.rp", g_ue8_rp, F32, -1},
    {"ue8m0x2.f32.rp.satfinite", g_ue8_rp_sat, F32, -1},
    {"ue8m0x2.bf16x2.rz", h_ue8_rz, BF16X2, -1},
    {"ue8m0x2.bf16x2.rz.satfinite", h_ue8_rz_sat, BF16X2, -1},
    {"ue8m0x2.bf16x2.rp", h_ue8_rp, BF16X2, -1},
    {"ue8m0x2.bf16x2.rp.satfinite", h_ue8_rp_sat, BF16X2, -1},
    {"bf16x2.ue8m0x2", i_bf16_ue8, UE8X2, -1},
    {"bf16x2.e4m3x2.scaled", j_e4m3, F8X2, SCALE},
    {"bf16x2.e4m3x2.scaled.relu", j_e4m3_relu, F8X2, SCALE},
    {"bf16x2.e4m3x2.scaled.satfinite", j_e4m3_sat, F8X2, SCALE},
    {"bf16x2.e4m3x2.scaled.relu.satfinite", j_e4m3_relu_sat, F8X2, SCALE},
    {"bf16x2.e5m2x2.scaled", k_e5m2, F8X2, SCALE},
    {"bf16x2.e5m2x2.scaled.relu", k_e5m2_relu, F8X2, SCALE},
    {"bf16x2.e5m2x2.scaled.satfinite", k_e5m2_sat, F8X2, SCALE},
    {"bf16x2.e5m2x2.scaled.relu.satfinite", k_e5m2_relu_sat, F8X2, SCALE},
    {"bf16x2.e2m1x2.scaled", l_e2m1, F4X2, SCALE},
    {"bf16x2.e2m1x2.scaled.relu", l_e2m1_relu, F4X2, SCALE},
    {"bf16x2.e2m1x2.scaled.satfinite", l_e2m1_sat, F4X2, SCALE},
    {"bf16x2.e2m1x2.scaled.relu.satfinite", l_e2m1_relu_sat, F4X2, SCALE},
};

static const int SECTION_COUNT = 8;

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
  std::vector<unsigned int> sections[SECTION_COUNT];
  for (int index = 0; index < SECTION_COUNT; ++index) sections[index] = read_section(input_file);
  fclose(input_file);

  cudaDeviceProp properties;
  CHECK(cudaGetDeviceProperties(&properties, 0));
  int driver_version = 0;
  CHECK(cudaDriverGetVersion(&driver_version));
  fprintf(stderr, "device=%s compute=%d.%d driver=%d\n", properties.name, properties.major,
          properties.minor, driver_version);

  size_t widest = 0;
  for (int index = 0; index < SECTION_COUNT; ++index)
    if (sections[index].size() > widest) widest = sections[index].size();
  unsigned int* device_primary = nullptr;
  unsigned int* device_secondary = nullptr;
  unsigned int* device_output = nullptr;
  CHECK(cudaMalloc(&device_primary, widest * sizeof(unsigned int)));
  CHECK(cudaMalloc(&device_secondary, widest * sizeof(unsigned int)));
  CHECK(cudaMalloc(&device_output, widest * sizeof(unsigned int)));

  FILE* output_file = fopen(argv[2], "wb");
  if (!output_file) {
    fprintf(stderr, "cannot open %s\n", argv[2]);
    return 1;
  }

  const int form_count = sizeof(FORM_ORDER) / sizeof(FORM_ORDER[0]);
  for (int index = 0; index < form_count; ++index) {
    const Form& form = FORM_ORDER[index];
    const std::vector<unsigned int>& primary = sections[form.primary];
    int count = (int)primary.size();
    CHECK(cudaMemcpy(device_primary, primary.data(), count * sizeof(unsigned int),
                     cudaMemcpyHostToDevice));
    if (form.secondary >= 0) {
      const std::vector<unsigned int>& secondary = sections[form.secondary];
      if ((int)secondary.size() != count) {
        fprintf(stderr, "form %s: section lengths disagree\n", form.name);
        return 1;
      }
      CHECK(cudaMemcpy(device_secondary, secondary.data(), count * sizeof(unsigned int),
                       cudaMemcpyHostToDevice));
    } else {
      CHECK(cudaMemset(device_secondary, 0, count * sizeof(unsigned int)));
    }
    CHECK(cudaMemset(device_output, 0, count * sizeof(unsigned int)));
    int threads = 128;
    int blocks = (count + threads - 1) / threads;
    form.kernel<<<blocks, threads>>>(device_output, device_primary, device_secondary, count);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<unsigned int> results(count);
    CHECK(cudaMemcpy(results.data(), device_output, count * sizeof(unsigned int),
                     cudaMemcpyDeviceToHost));
    unsigned int written = (unsigned int)count;
    fwrite(&written, sizeof(written), 1, output_file);
    fwrite(results.data(), sizeof(unsigned int), count, output_file);
    fprintf(stderr, "form %-40s n=%d\n", form.name, count);
  }
  fclose(output_file);
  return 0;
}
