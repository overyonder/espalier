// Match a fallback atlas to foreground colour statistics in aligned painted
// views.
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <png.h>
#include <stdexcept>
#include <vector>

namespace {
struct Image {
  unsigned width, height;
  std::vector<unsigned char> rgba;
};
Image load(const std::filesystem::path &path) {
  png_image png{};
  png.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_file(&png, path.c_str()))
    throw std::runtime_error("Cannot read " + path.string());
  png.format = PNG_FORMAT_RGBA;
  Image image{png.width, png.height,
              std::vector<unsigned char>(PNG_IMAGE_SIZE(png))};
  if (!png_image_finish_read(&png, nullptr, image.rgba.data(), 0, nullptr)) {
    const std::string error = png.message;
    png_image_free(&png);
    throw std::runtime_error(error);
  }
  png_image_free(&png);
  return image;
}
struct Moments {
  double weight = 0;
  std::array<double, 3> sum{}, squares{};
  void add(const unsigned char *pixel, double w) {
    weight += w;
    for (unsigned c = 0; c < 3; ++c) {
      const double value = pixel[c] / 255.;
      sum[c] += w * value;
      squares[c] += w * value * value;
    }
  }
};
bool foreground(const unsigned char *pixel) {
  return pixel[3] >= 128 && std::max({pixel[0], pixel[1], pixel[2]}) > 4;
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 7)
      throw std::runtime_error("Usage: match-base-colour BASE.png CAMERAS.json "
                               "RENDERS PAINTINGS OUTPUT.png REPORT.toml");
    std::ifstream input(argv[2]);
    const auto cameras = nlohmann::json::parse(input).at("cams");
    if (cameras.empty())
      throw std::runtime_error("No calibration views");
    Moments source, target;
    unsigned used_views = 0;
    for (const auto &camera : cameras) {
      const auto name = camera.at("name").get<std::string>() + ".png";
      const auto rendered = load(std::filesystem::path(argv[3]) / name);
      const auto painted = load(std::filesystem::path(argv[4]) / name);
      // Equal camera weighting, using corresponding foreground samples on both
      // canvases.
      constexpr unsigned grid = 256;
      std::vector<std::array<size_t, 2>> pairs;
      for (unsigned y = 0; y < grid; ++y)
        for (unsigned x = 0; x < grid; ++x) {
          const size_t a =
              4 * (size_t((2 * y + 1) * rendered.height / (2 * grid)) *
                       rendered.width +
                   (2 * x + 1) * rendered.width / (2 * grid));
          const size_t b =
              4 * (size_t((2 * y + 1) * painted.height / (2 * grid)) *
                       painted.width +
                   (2 * x + 1) * painted.width / (2 * grid));
          if (foreground(&rendered.rgba[a]) && foreground(&painted.rgba[b]))
            pairs.push_back({a, b});
        }
      if (pairs.size() < 64)
        continue;
      ++used_views;
      const double weight = 1. / pairs.size();
      for (const auto pair : pairs) {
        source.add(&rendered.rgba[pair[0]], weight);
        target.add(&painted.rgba[pair[1]], weight);
      }
    }
    if (!used_views)
      throw std::runtime_error(
          "No overlapping painted foreground for colour matching");
    std::array<double, 3> gain{}, offset{};
    for (unsigned c = 0; c < 3; ++c) {
      const double a = source.sum[c] / source.weight,
                   b = target.sum[c] / target.weight;
      const double av = std::max(0., source.squares[c] / source.weight - a * a);
      const double bv = std::max(0., target.squares[c] / target.weight - b * b);
      // A flat source channel has no contrast to rescale: match its mean only.
      gain[c] = av > 1e-6 ? std::sqrt(bv / av) : 1.;
      offset[c] = b - gain[c] * a;
    }
    auto atlas = load(argv[1]);
    size_t clipped = 0;
    for (size_t i = 0; i < atlas.rgba.size(); i += 4) {
      if (!atlas.rgba[i + 3])
        continue;
      for (unsigned c = 0; c < 3; ++c) {
        const double value = gain[c] * atlas.rgba[i + c] / 255. + offset[c];
        clipped += value < 0 || value > 1;
        atlas.rgba[i + c] = static_cast<unsigned char>(
            std::lround(std::clamp(value, 0., 1.) * 255));
      }
    }
    png_image output{};
    output.version = PNG_IMAGE_VERSION;
    output.width = atlas.width;
    output.height = atlas.height;
    output.format = PNG_FORMAT_RGBA;
    if (!png_image_write_to_file(&output, argv[5], 0, atlas.rgba.data(), 0,
                                 nullptr))
      throw std::runtime_error(output.message);
    std::ofstream report(argv[6]);
    report << "calibration_views = " << used_views << "\ngain = [" << gain[0]
           << ", " << gain[1] << ", " << gain[2] << "]\noffset = [" << offset[0]
           << ", " << offset[1] << ", " << offset[2]
           << "]\nclipped_channels = " << clipped << '\n';
    if (!report)
      throw std::runtime_error("Cannot write colour match report");
    std::cout << "BASE_COLOUR_MATCHED " << used_views << " views\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
