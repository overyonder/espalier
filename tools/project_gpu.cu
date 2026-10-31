// Resident projection painter. Blender owns mesh/image I/O; CUDA owns
// visibility, sampling, blending and padding.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

struct Projection {
  unsigned size;
  size_t count;
  float4 *positions, *normals, *colour, *scratch;
  float *best;
  unsigned char *frozen, *filled, *next_filled;
  unsigned *counts;
  float4 *view, *view_positions;
  size_t view_capacity, position_capacity;
};
struct Camera {
  double values[25];
};
static thread_local char error_message[256];
static bool checked(cudaError_t status) {
  if (status == cudaSuccess)
    return true;
  snprintf(error_message, sizeof(error_message), "%s",
           cudaGetErrorString(status));
  return false;
}
extern "C" const char *projection_error() { return error_message; }
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    if (!checked(call))                                                        \
      return -1;                                                               \
  } while (0)

__device__ float clamp01(float x) { return fminf(1.f, fmaxf(0.f, x)); }
__device__ double clip(double x, double lo, double hi) {
  return fmin(hi, fmax(lo, x));
}
__device__ float linear(float c) {
  return c <= .04045f ? c / 12.92f : powf((c + .055f) / 1.055f, 2.4f);
}
__device__ double linear_sample(double c) {
  return c <= .04045 ? c / 12.92 : pow((c + .055) / 1.055, 2.4);
}
__device__ float srgb(float c) {
  return c <= .0031308f ? c * 12.92f
                        : 1.055f * powf(fmaxf(c, 0.f), 1.f / 2.4f) - .055f;
}

__global__ void prepare_surface(Projection p, double scale, double sx,
                                double sy, double sz) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count)
    return;
  float4 a = p.positions[i], n = p.normals[i];
  bool valid = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z) > .1f;
  p.positions[i] =
      make_float4((float)((a.x - sx) / scale), (float)((a.y - sy) / scale),
                  (float)((a.z - sz) / scale), valid ? 1.f : 0.f);
  n.x = n.x * 2.f - 1.f;
  n.y = n.y * 2.f - 1.f;
  n.z = n.z * 2.f - 1.f;
  float length = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z) + 1.e-9f;
  p.normals[i] = make_float4(n.x / length, n.y / length, n.z / length, 0.f);
}
__device__ float channel(float4 c, unsigned k) {
  return k == 0 ? c.x : k == 1 ? c.y : c.z;
}
__global__ void paint_surface(Projection p, Camera cam, unsigned generated_size,
                              unsigned position_size) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count || p.positions[i].w == 0.f)
    return;
  const double *c = cam.values;
  float4 point = p.positions[i], normal = p.normals[i];
  double x = c[0] * point.x + c[1] * point.y + c[2] * point.z + c[3];
  double y = c[4] * point.x + c[5] * point.y + c[6] * point.z + c[7];
  double z = -(c[8] * point.x + c[9] * point.y + c[10] * point.z + c[11]);
  double u = (c[15] * x / fmax(z, 1.e-6) - c[16] + .5) * position_size;
  double v = (.5 - (c[15] * y / fmax(z, 1.e-6) - c[17])) * position_size;
  if (z <= 0 || u < 0 || v < 0 || u >= position_size - 1 ||
      v >= position_size - 1)
    return;
  float4 hit = p.view_positions[(unsigned)v * position_size + (unsigned)u];
  float dx = (float)((hit.x - c[21]) / c[20]) - point.x;
  float dy = (float)((hit.y - c[22]) / c[20]) - point.y;
  float dz = (float)((hit.z - c[23]) / c[20]) - point.z;
  if (hit.w <= .5f || sqrtf(dx * dx + dy * dy + dz * dz) >= c[19])
    return;
  atomicAdd(p.counts, 1);
  if (p.frozen[i])
    return;
  double vx = c[12] - point.x, vy = c[13] - point.y, vz = c[14] - point.z;
  double distance = sqrt(vx * vx + vy * vy + vz * vz);
  double cosine = clip(normal.x * (vx / distance) + normal.y * (vy / distance) +
                           normal.z * (vz / distance),
                       0., 1.);
  double edge = clip(
      fmin(fmin(u, v), fmin(position_size - 1 - u, position_size - 1 - v)) /
          (.08 * position_size),
      0., 1.);
  float weight = (float)(pow(cosine, 6.) * edge), previous = p.best[i];
  p.best[i] = fmaxf(previous, weight);
  if (weight <= 1.e-4f)
    return;
  double scale = (double)generated_size / position_size, ug = u * scale,
         vg = v * scale;
  int u0 = (int)clip(floor(ug), 0., generated_size - 2),
      v0 = (int)clip(floor(vg), 0., generated_size - 2);
  double fu = ug - u0, fv = vg - v0;
  float4 a = p.view[v0 * generated_size + u0],
         b = p.view[v0 * generated_size + u0 + 1];
  float4 d = p.view[(v0 + 1) * generated_size + u0],
         e = p.view[(v0 + 1) * generated_size + u0 + 1];
  float wa = powf(weight, (float)c[24]), wb = powf(previous, (float)c[24]);
  float alpha = wa / (wa + wb + 1.e-12f);
  float4 old = p.colour[i], result = old;
  float *channels = &result.x;
  for (unsigned k = 0; k < 3; ++k) {
    double sample = channel(a, k) * (1. - fu) * (1. - fv) +
                    channel(b, k) * fu * (1. - fv) +
                    channel(d, k) * (1. - fu) * fv + channel(e, k) * fu * fv;
    float accumulated = (float)(linear_sample(sample) * weight);
    float updated = accumulated / fmaxf(weight, 1.e-6f) * alpha +
                    linear(clamp01(channel(old, k))) * (1.f - alpha);
    channels[k] = srgb(updated);
  }
  p.colour[i] = result;
}
__global__ void start_padding(Projection p) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.count)
    p.filled[i] = p.positions[i].w != 0.f;
}
__global__ void pad_surface(Projection p) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count)
    return;
  p.scratch[i] = p.colour[i];
  p.next_filled[i] = p.filled[i];
  if (p.filled[i])
    return;
  unsigned x = i % p.size, y = i / p.size;
  size_t neighbours[4] = {(size_t)y * p.size + (x ? x - 1 : p.size - 1),
                          (size_t)y * p.size + (x + 1 == p.size ? 0 : x + 1),
                          (size_t)(y ? y - 1 : p.size - 1) * p.size + x,
                          (size_t)(y + 1 == p.size ? 0 : y + 1) * p.size + x};
  float4 sum = make_float4(0, 0, 0, 1);
  unsigned count = 0;
  for (unsigned j = 0; j < 4; ++j)
    if (p.filled[neighbours[j]]) {
      float4 n = p.colour[neighbours[j]];
      sum.x += n.x;
      sum.y += n.y;
      sum.z += n.z;
      ++count;
    }
  if (count) {
    p.scratch[i] = make_float4(sum.x / count, sum.y / count, sum.z / count, 1);
    p.next_filled[i] = 1;
  }
}
__global__ void quantize_surface(Projection p) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count)
    return;
  float4 c = p.colour[i];
  c.x = floorf(clamp01(c.x) * 255.f + .5f) * (1.f / 255.f);
  c.y = floorf(clamp01(c.y) * 255.f + .5f) * (1.f / 255.f);
  c.z = floorf(clamp01(c.z) * 255.f + .5f) * (1.f / 255.f);
  c.w = 1.f;
  p.colour[i] = c;
  if (p.positions[i].w) {
    atomicAdd(p.counts + 1, 1);
    if (p.best[i] > 1.e-4f || p.frozen[i])
      atomicAdd(p.counts + 2, 1);
  }
}
extern "C" void projection_destroy(Projection *p) {
  if (!p)
    return;
  cudaFree(p->positions);
  cudaFree(p->normals);
  cudaFree(p->colour);
  cudaFree(p->scratch);
  cudaFree(p->best);
  cudaFree(p->frozen);
  cudaFree(p->filled);
  cudaFree(p->next_filled);
  cudaFree(p->counts);
  cudaFree(p->view);
  cudaFree(p->view_positions);
  free(p);
}
extern "C" Projection *projection_create(unsigned size, const float *positions,
                                         const float *normals,
                                         const float *colour, const float *best,
                                         const unsigned char *frozen,
                                         double scale, double sx, double sy,
                                         double sz) {
  Projection *p = (Projection *)calloc(1, sizeof(*p));
  if (!p)
    return nullptr;
  p->size = size;
  p->count = (size_t)size * size;
  bool ok =
      checked(cudaMalloc(&p->positions, p->count * sizeof(float4))) &&
      checked(cudaMalloc(&p->normals, p->count * sizeof(float4))) &&
      checked(cudaMalloc(&p->colour, p->count * sizeof(float4))) &&
      checked(cudaMalloc(&p->scratch, p->count * sizeof(float4))) &&
      checked(cudaMalloc(&p->best, p->count * sizeof(float))) &&
      checked(cudaMalloc(&p->frozen, p->count)) &&
      checked(cudaMalloc(&p->filled, p->count)) &&
      checked(cudaMalloc(&p->next_filled, p->count)) &&
      checked(cudaMalloc(&p->counts, 3 * sizeof(unsigned))) &&
      checked(cudaMemcpy(p->positions, positions, p->count * sizeof(float4),
                         cudaMemcpyHostToDevice)) &&
      checked(cudaMemcpy(p->normals, normals, p->count * sizeof(float4),
                         cudaMemcpyHostToDevice)) &&
      checked(cudaMemcpy(p->colour, colour, p->count * sizeof(float4),
                         cudaMemcpyHostToDevice)) &&
      checked(cudaMemcpy(p->best, best, p->count * sizeof(float),
                         cudaMemcpyHostToDevice)) &&
      checked(cudaMemcpy(p->frozen, frozen, p->count, cudaMemcpyHostToDevice));
  if (!ok) {
    projection_destroy(p);
    return nullptr;
  }
  prepare_surface<<<(p->count + 255) / 256, 256>>>(*p, scale, sx, sy, sz);
  if (!checked(cudaDeviceSynchronize())) {
    projection_destroy(p);
    return nullptr;
  }
  return p;
}
extern "C" int projection_paint(Projection *p, const double *camera,
                                const float *view, unsigned generated_size,
                                const float *positions,
                                unsigned position_size) {
  size_t bytes = (size_t)generated_size * generated_size * sizeof(float4),
         pbytes = (size_t)position_size * position_size * sizeof(float4);
  if (bytes > p->view_capacity) {
    CUDA_CHECK(cudaFree(p->view));
    CUDA_CHECK(cudaMalloc(&p->view, bytes));
    p->view_capacity = bytes;
  }
  if (pbytes > p->position_capacity) {
    CUDA_CHECK(cudaFree(p->view_positions));
    CUDA_CHECK(cudaMalloc(&p->view_positions, pbytes));
    p->position_capacity = pbytes;
  }
  CUDA_CHECK(cudaMemcpy(p->view, view, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(p->view_positions, positions, pbytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(p->counts, 0, 3 * sizeof(unsigned)));
  Camera cam;
  for (unsigned j = 0; j < 25; ++j)
    cam.values[j] = camera[j];
  unsigned blocks = (p->count + 255) / 256;
  paint_surface<<<blocks, 256>>>(*p, cam, generated_size, position_size);
  start_padding<<<blocks, 256>>>(*p);
  for (unsigned j = 0; j < 12; ++j) {
    pad_surface<<<blocks, 256>>>(*p);
    float4 *colour = p->colour;
    p->colour = p->scratch;
    p->scratch = colour;
    unsigned char *filled = p->filled;
    p->filled = p->next_filled;
    p->next_filled = filled;
  }
  quantize_surface<<<blocks, 256>>>(*p);
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
extern "C" int projection_download(Projection *p, float *colour, float *best,
                                   unsigned *counts) {
  CUDA_CHECK(cudaMemcpy(colour, p->colour, p->count * sizeof(float4),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(best, p->best, p->count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(counts, p->counts, 3 * sizeof(unsigned),
                        cudaMemcpyDeviceToHost));
  return 0;
}
