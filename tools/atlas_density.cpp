// Atlas density against the sharpest eligible camera painting, plus UV
// occupancy.
#include "painted_mesh.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <png.h>
#include <stdexcept>
#include <vector>

namespace {
using Json = nlohmann::json;
struct Vector3 {
  double x, y, z;
  Vector3 operator+(Vector3 b) const { return {x + b.x, y + b.y, z + b.z}; }
  Vector3 operator-(Vector3 b) const { return {x - b.x, y - b.y, z - b.z}; }
  Vector3 operator*(double s) const { return {x * s, y * s, z * s}; }
};
struct Vector2 {
  double x, y;
  Vector2 operator-(Vector2 b) const { return {x - b.x, y - b.y}; }
};
double dot(Vector3 a, Vector3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double cross(Vector2 a, Vector2 b) { return a.x * b.y - a.y * b.x; }
Vector3 cross(Vector3 a, Vector3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
struct Vertex {
  Vector3 position, normal;
  Vector2 uv;
};
struct Camera {
  Vector3 origin, right, up, back;
  double focal, shift_x, shift_y;
  int painting_size;
};
struct Projection {
  Vector2 pixel;
  double inverse_depth;
};
void require(bool valid, const char *message) {
  if (!valid)
    throw std::runtime_error(message);
}
Vector3 camera_local(const Camera &c, Vector3 p) {
  auto d = p - c.origin;
  return {dot(c.right, d), dot(c.up, d), dot(c.back, d)};
}
Projection project(const Camera &c, Vector3 p, int size) {
  auto q = camera_local(c, p);
  return {{(c.focal * q.x / -q.z + 2 * c.shift_x + 1) * size * .5,
           (c.focal * q.y / -q.z + 2 * c.shift_y + 1) * size * .5},
          1 / -q.z};
}
uint32_t read_u32(const std::vector<uint8_t> &data, size_t offset) {
  require(offset <= data.size() && data.size() - offset >= 4, "Truncated GLB");
  uint32_t value;
  std::memcpy(&value, data.data() + offset, 4);
  return value;
}
std::array<int, 2> png_dimensions(const std::filesystem::path &path) {
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  require(png_image_begin_read_from_file(&image, path.c_str()),
          "Cannot read painting PNG");
  const std::array<int, 2> size{int(image.width), int(image.height)};
  png_image_free(&image);
  return size;
}
int final_atlas_size(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(bool(input), "Cannot open final model");
  auto length = input.tellg();
  require(length >= 28, "Truncated final model");
  std::vector<uint8_t> data(static_cast<size_t>(length));
  input.seekg(0);
  input.read(reinterpret_cast<char *>(data.data()), length);
  require(bool(input), "Cannot read final model");
  require(read_u32(data, 0) == 0x46546c67 && read_u32(data, 4) == 2 &&
              read_u32(data, 8) == data.size(),
          "Expected GLB 2");
  auto json_length = read_u32(data, 12);
  require(read_u32(data, 16) == 0x4e4f534a &&
              20ull + json_length + 8 <= data.size(),
          "Invalid GLB JSON");
  auto json = Json::parse(data.begin() + 20, data.begin() + 20 + json_length);
  size_t binary = 28 + json_length;
  require(read_u32(data, binary - 4) == 0x004e4942, "Missing GLB binary chunk");
  auto texture = json.at("materials")
                     .at(0)
                     .at("pbrMetallicRoughness")
                     .at("baseColorTexture")
                     .at("index");
  auto image_index =
      json.at("textures").at(texture.get<size_t>()).at("source").get<size_t>();
  auto &image = json.at("images").at(image_index);
  require(image.at("mimeType") == "image/png", "Expected packaged PNG albedo");
  auto &view = json.at("bufferViews").at(image.at("bufferView").get<size_t>());
  size_t offset = binary + view.value("byteOffset", size_t(0)),
         size = view.at("byteLength");
  require(view.at("buffer") == 0 && offset <= data.size() &&
              size <= data.size() - offset,
          "Invalid albedo buffer view");
  png_image png{};
  png.version = PNG_IMAGE_VERSION;
  require(png_image_begin_read_from_memory(&png, data.data() + offset, size),
          "Invalid albedo PNG");
  auto width = png.width, height = png.height;
  png_image_free(&png);
  require(width == height && width > 0 && width <= 16384,
          "Density report expects a square atlas up to 16384");
  return int(width);
}
// UV sample centres and the baker's four-texel padding, measured geometrically.
template <class F>
void raster_triangle(Vector2 a, Vector2 b, Vector2 c, int size, int padding,
                     F visit) {
  double determinant = cross(b - a, c - a);
  if (std::abs(determinant) < 1e-12)
    return;
  int x0 =
      std::max(0, int(std::ceil(std::min({a.x, b.x, c.x}) - .5 - padding)));
  int x1 = std::min(size - 1,
                    int(std::floor(std::max({a.x, b.x, c.x}) - .5 + padding)));
  int y0 =
      std::max(0, int(std::ceil(std::min({a.y, b.y, c.y}) - .5 - padding)));
  int y1 = std::min(size - 1,
                    int(std::floor(std::max({a.y, b.y, c.y}) - .5 + padding)));
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) {
      Vector2 sample{x + .5, y + .5};
      double v = cross(sample - a, c - a) / determinant,
             w = cross(b - a, sample - a) / determinant;
      bool inside = v >= -1e-6 && w >= -1e-6 && v + w <= 1.000001;
      if (!inside) {
        if (!padding)
          continue;
        double closest = std::numeric_limits<double>::infinity();
        std::array<Vector2, 3> corners{a, b, c};
        for (int i = 0; i < 3; ++i) {
          auto start = corners[i], delta = corners[(i + 1) % 3] - start,
               relative = sample - start;
          double t = std::clamp(
              (relative.x * delta.x + relative.y * delta.y) /
                  std::max(delta.x * delta.x + delta.y * delta.y, 1e-20),
              0., 1.);
          Vector2 distance{relative.x - t * delta.x, relative.y - t * delta.y};
          closest = std::min(closest,
                             distance.x * distance.x + distance.y * distance.y);
        }
        if (closest > padding * padding)
          continue;
      }
      visit(x, y, v, w, inside);
    }
}
void report_density(const std::filesystem::path &bake,
                    const std::filesystem::path &cameras_path,
                    const std::filesystem::path &paintings,
                    const std::filesystem::path &model,
                    const std::filesystem::path &prefix) {
  auto mesh = painted_mesh::load(bake / "sparse-uv.ply");
  int size = final_atlas_size(model);
  std::ifstream timing_input(bake / "bake-timings.json");
  auto timings = Json::parse(timing_input);
  auto normal_mode = timings.at("normal_mode").get<std::string>();
  require(normal_mode == "stored_vertex" || normal_mode == "implicit_flat",
          "Unknown baker normal mode");
  std::vector<Vertex> vertices(mesh.vertex_bytes.size() / 32);
  for (size_t i = 0; i < vertices.size(); ++i) {
    float a[8];
    std::memcpy(a, mesh.vertex_bytes.data() + i * 32, 32);
    require(a[6] >= 0 && a[6] <= 1 && a[7] >= 0 && a[7] <= 1,
            "UV coordinates outside the packed atlas");
    vertices[i] = {
        {a[0], -a[2], a[1]}, {a[3], -a[5], a[4]}, {a[6] * size, a[7] * size}};
  }
  std::ifstream camera_input(cameras_path);
  auto camera_data = Json::parse(camera_input);
  int resolution = camera_data.at("res");
  require(resolution > 0 && resolution <= 16384,
          "Invalid visibility resolution");
  std::vector<Camera> cameras;
  for (auto &camera : camera_data.at("cams")) {
    auto name = camera.at("name").get<std::string>();
    auto dimensions = png_dimensions(paintings / (name + ".png"));
    require(dimensions[0] > 0 && dimensions[0] == dimensions[1],
            "Painting must preserve its square camera canvas");
    auto &m = camera.at("matrix");
    cameras.push_back({{m[0][3], m[1][3], m[2][3]},
                       {m[0][0], m[1][0], m[2][0]},
                       {m[0][1], m[1][1], m[2][1]},
                       {m[0][2], m[1][2], m[2][2]},
                       2 * camera.at("lens").get<double>() /
                           camera.at("sensor").get<double>(),
                       camera.at("shift")[0],
                       camera.at("shift")[1],
                       dimensions[0]});
  }
  require(!cameras.empty() && cameras.size() <= 16,
          "Expected one to sixteen cameras");
  size_t depth_pixels = size_t(resolution) * resolution;
  std::vector<std::vector<float>> depths(cameras.size(),
                                         std::vector<float>(depth_pixels, 0));
  for (size_t k = 0; k < cameras.size(); ++k) {
    for (auto triangle : mesh.triangles) {
      auto a = project(cameras[k], vertices[triangle[0]].position, resolution),
           b = project(cameras[k], vertices[triangle[1]].position, resolution),
           c = project(cameras[k], vertices[triangle[2]].position, resolution);
      require(std::min({a.inverse_depth, b.inverse_depth, c.inverse_depth}) > 0,
              "Saved camera intersects receiving mesh");
      raster_triangle(a.pixel, b.pixel, c.pixel, resolution, 0,
                      [&](int x, int y, double v, double w, bool) {
                        float depth =
                            float(a.inverse_depth * (1 - v - w) +
                                  b.inverse_depth * v + c.inverse_depth * w);
                        auto &stored = depths[k][size_t(y) * resolution + x];
                        stored = std::max(stored, depth);
                      });
    }
    std::cout << "DENSITY_VISIBILITY " << k + 1 << "/" << cameras.size()
              << std::endl;
  }
  size_t texels = size_t(size) * size;
  std::vector<float> density(texels, std::numeric_limits<float>::infinity());
  // 0: outside the padded charts; 1: current four-texel bake gutter; 2:
  // triangle interior.
  std::vector<uint8_t> occupancy(texels, 0);
  size_t face_index = 0;
  for (auto triangle : mesh.triangles) {
    auto a = vertices[triangle[0]], b = vertices[triangle[1]],
         c = vertices[triangle[2]];
    auto ab = b.uv - a.uv, ac = c.uv - a.uv;
    double determinant = cross(ab, ac);
    if (std::abs(determinant) < 1e-12)
      continue;
    auto du =
        ((b.position - a.position) * ac.y - (c.position - a.position) * ab.y) *
        (1 / determinant);
    auto dv =
        ((c.position - a.position) * ab.x - (b.position - a.position) * ac.x) *
        (1 / determinant);
    auto flat_normal = cross(b.position - a.position, c.position - a.position);
    std::vector<std::array<Projection, 3>> projected;
    for (auto &camera : cameras)
      projected.push_back({project(camera, a.position, resolution),
                           project(camera, b.position, resolution),
                           project(camera, c.position, resolution)});
    raster_triangle(
        a.uv, b.uv, c.uv, size, 4,
        [&](int x, int y, double v, double w, bool inside) {
          size_t index = size_t(y) * size + x;
          if (!inside) {
            if (!occupancy[index])
              occupancy[index] = 1;
            return;
          }
          if (occupancy[index] == 2)
            return;
          occupancy[index] = 2;
          auto point =
              a.position * (1 - v - w) + b.position * v + c.position * w;
          auto normal =
              normal_mode == "implicit_flat"
                  ? flat_normal
                  : a.normal * (1 - v - w) + b.normal * v + c.normal * w;
          double normal_length = std::sqrt(dot(normal, normal));
          if (normal_length < 1e-12)
            return;
          for (size_t k = 0; k < cameras.size(); ++k) {
            auto &camera = cameras[k];
            auto ray = camera.origin - point;
            double cosine =
                dot(normal, ray) / (normal_length * std::sqrt(dot(ray, ray)));
            if (cosine < .17364818)
              continue;
            auto projected_point = project(camera, point, resolution);
            int px = int(std::floor(projected_point.pixel.x)),
                py = int(std::floor(projected_point.pixel.y));
            if (px < 0 || py < 0 || px >= resolution || py >= resolution ||
                projected_point.inverse_depth <= 0)
              continue;
            auto t = projected[k];
            double screen_det =
                cross(t[1].pixel - t[0].pixel, t[2].pixel - t[0].pixel);
            if (std::abs(screen_det) < 1e-12)
              continue;
            Vector2 sample{px + .5, py + .5};
            double sv = cross(sample - t[0].pixel, t[2].pixel - t[0].pixel) /
                        screen_det,
                   sw = cross(t[1].pixel - t[0].pixel, sample - t[0].pixel) /
                        screen_det;
            double inverse_depth = t[0].inverse_depth * (1 - sv - sw) +
                                   t[1].inverse_depth * sv +
                                   t[2].inverse_depth * sw;
            float seen = depths[k][size_t(py) * resolution + px];
            if (seen <= 0 || std::abs(1 / inverse_depth - 1 / seen) > .0015)
              continue;
            auto q = camera_local(camera, point);
            Vector3 u{dot(camera.right, du), dot(camera.up, du),
                      dot(camera.back, du)},
                vv{dot(camera.right, dv), dot(camera.up, dv),
                   dot(camera.back, dv)};
            double factor =
                camera.focal * camera.painting_size * .5 / (q.z * q.z);
            Vector2 pu{factor * (-u.x * q.z + q.x * u.z),
                       factor * (-u.y * q.z + q.y * u.z)},
                pv{factor * (-vv.x * q.z + q.x * vv.z),
                   factor * (-vv.y * q.z + q.y * vv.z)};
            double area = std::abs(cross(pu, pv));
            if (area > 1e-20)
              density[index] =
                  std::min(density[index], float(1 / std::sqrt(area)));
          }
        });
    if (++face_index % 10000 == 0)
      std::cout << "DENSITY_TRIANGLES " << face_index << "/"
                << mesh.triangles.size() << std::endl;
  }
  std::vector<uint8_t> rgb(texels * 3);
  std::vector<float> values;
  size_t surface = 0, gutter = 0, unseen = 0;
  const std::array<std::array<double, 3>, 6> colors{{{25, 80, 190},
                                                     {40, 180, 205},
                                                     {60, 190, 95},
                                                     {245, 210, 45},
                                                     {235, 105, 30},
                                                     {190, 35, 50}}};
  for (size_t i = 0; i < texels; ++i) {
    if (!occupancy[i])
      continue;
    if (occupancy[i] == 1) {
      ++gutter;
      rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = 72;
      continue;
    }
    ++surface;
    if (!std::isfinite(density[i])) {
      ++unseen;
      rgb[i * 3] = rgb[i * 3 + 2] = 220;
      continue;
    }
    values.push_back(density[i]);
    double t = std::clamp(std::log2(density[i]) + 1., 0., 5.);
    int low = std::min(4, int(t));
    double fraction = t - low;
    for (int channel = 0; channel < 3; ++channel)
      rgb[i * 3 + channel] =
          uint8_t(std::lround(colors[low][channel] * (1 - fraction) +
                              colors[low + 1][channel] * fraction));
  }
  require(surface > 0, "No atlas texel centres inside UV triangles");
  auto image_path = prefix.string() + ".png",
       stats_path = prefix.string() + ".toml";
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  image.width = size;
  image.height = size;
  image.format = PNG_FORMAT_RGB;
  require(png_image_write_to_file(&image, image_path.c_str(), 0, rgb.data(), 0,
                                  nullptr),
          "Cannot write density PNG");
  std::ofstream stats(stats_path);
  stats << "atlas_size = " << size << "\ncamera_count = " << cameras.size()
        << "\nsurface_texels = " << surface << "\ngutter_texels = " << gutter
        << "\nunassigned_texels = " << texels - surface - gutter
        << "\nno_eligible_view_texels = " << unseen
        << "\nuv_occupancy_percent = " << 100. * surface / texels
        << "\ngutter_percent = " << 100. * gutter / texels
        << "\nunassigned_percent = "
        << 100. * (texels - surface - gutter) / texels << "\n";
  stats << "\n[legend]\nblue = 0.5\ncyan = 1.0\ngreen = 2.0\nyellow = "
           "4.0\norange = 8.0\nred = 16.0\ngrey = 'four-texel gutter'\nblack = "
           "'outside padded charts'\nmagenta = 'no eligible view'\n";
  if (!values.empty()) {
    std::sort(values.begin(), values.end());
    stats << "\n[atlas_texels_per_source_pixel]\np05 = "
          << values[size_t(.05 * (values.size() - 1))]
          << "\nmedian = " << values[values.size() / 2]
          << "\np95 = " << values[size_t(.95 * (values.size() - 1))] << "\n";
  }
  stats.flush();
  require(bool(stats), "Cannot write density statistics");
  std::cout << "DENSITY_MAP " << image_path << "\nUV occupancy "
            << 100. * surface / texels << "%; four-texel gutter "
            << 100. * gutter / texels << "%; unassigned "
            << 100. * (texels - surface - gutter) / texels << "%\n";
  std::cout << "Density: atlas texels per source pixel, area-equivalent linear "
               "ratio, sharpest eligible view. Blue 0.5; cyan 1; green 2; "
               "yellow 4; orange 8; red 16+. Grey gutter; black unassigned; "
               "magenta no eligible view.\n";
}
} // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 6, "Usage: atlas-density BAKE_DIRECTORY CAMERAS.json "
                       "PAINTINGS FINAL.glb OUTPUT_PREFIX");
    report_density(argv[1], argv[2], argv[3], argv[4], argv[5]);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
