#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace painted_mesh {

// Resident baker layout: position, normal, UV; eight little-endian floats.
struct Mesh {
  std::vector<uint8_t> vertex_bytes;
  std::vector<std::array<uint32_t, 3>> triangles;
};

inline Mesh load(const std::filesystem::path &path) {
  static_assert(std::endian::native == std::endian::little);
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("Cannot open painted PLY");
  std::string line;
  const auto expect = [&](const std::string &expected) {
    if (!std::getline(input, line) || line != expected)
      throw std::runtime_error("PLY does not match resident baker schema");
  };
  const auto count = [&](const std::string &prefix) {
    if (!std::getline(input, line) || !line.starts_with(prefix))
      throw std::runtime_error("Missing PLY element count");
    size_t consumed = 0;
    const auto value = std::stoull(line.substr(prefix.size()), &consumed);
    if (!value || value >= UINT32_MAX ||
        consumed != line.size() - prefix.size())
      throw std::runtime_error("Invalid PLY element count");
    return static_cast<size_t>(value);
  };
  expect("ply");
  expect("format binary_little_endian 1.0");
  const size_t vertices = count("element vertex ");
  for (const char *name : {"x", "y", "z", "nx", "ny", "nz", "s", "t"})
    expect(std::string("property float ") + name);
  const size_t faces = count("element face ");
  expect("property list uchar int vertex_indices");
  expect("end_header");
  const auto payload_start = input.tellg();
  const auto file_size = std::filesystem::file_size(path);
  if (payload_start < 0 || static_cast<uint64_t>(payload_start) > file_size ||
      file_size - static_cast<uint64_t>(payload_start) !=
          vertices * uint64_t(32) + faces * uint64_t(13))
    throw std::runtime_error("Unexpected painted PLY payload size");
  Mesh mesh;
  mesh.vertex_bytes.resize(vertices * 32);
  input.read(reinterpret_cast<char *>(mesh.vertex_bytes.data()),
             mesh.vertex_bytes.size());
  if (!input)
    throw std::runtime_error("Truncated PLY vertices");
  for (size_t offset = 0; offset < mesh.vertex_bytes.size(); offset += 4) {
    float value;
    std::memcpy(&value, mesh.vertex_bytes.data() + offset, 4);
    if (!std::isfinite(value))
      throw std::runtime_error("Nonfinite PLY attribute");
  }
  mesh.triangles.resize(faces);
  for (auto &triangle : mesh.triangles) {
    if (input.get() != 3)
      throw std::runtime_error("Expected PLY triangle");
    input.read(reinterpret_cast<char *>(triangle.data()), 12);
    if (!input)
      throw std::runtime_error("Truncated PLY triangle");
    for (auto index : triangle)
      if (index >= vertices)
        throw std::runtime_error("Invalid PLY vertex index");
  }
  if (input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("Unexpected data after PLY faces");
  return mesh;
}

} // namespace painted_mesh
