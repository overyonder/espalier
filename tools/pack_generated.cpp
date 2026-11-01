// Package the resident baker's exact final PLY and BC7 KTX1, without remeshing.
#include "painted_mesh.h"
#include <nlohmann/json.hpp>
#include <png.h>
#define BCDEC_IMPLEMENTATION
#include <bcdec.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Bytes = std::vector<uint8_t>;
using Json = nlohmann::json;

static void require(bool valid, const char *message) {
  if (!valid)
    throw std::runtime_error(message);
}

static void
remap_texture_references(Json &value,
                         const std::function<size_t(size_t)> &texture_index,
                         const std::string &key = "") {
  if (value.is_object()) {
    if (key.ends_with("Texture") && value.contains("index"))
      value["index"] = texture_index(value["index"].get<size_t>());
    for (auto &[name, child] : value.items())
      remap_texture_references(child, texture_index, name);
  } else if (value.is_array()) {
    for (auto &child : value)
      remap_texture_references(child, texture_index);
  }
}

static uint32_t u32(const Bytes &bytes, size_t offset) {
  require(offset <= bytes.size() && bytes.size() - offset >= 4,
          "Truncated KTX1 field");
  uint32_t value;
  std::memcpy(&value, bytes.data() + offset, 4);
  return value;
}

static Bytes read_file(const char *path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(input.good(), "Cannot open input");
  const auto size = input.tellg();
  require(size >= 0, "Cannot measure input");
  Bytes bytes(static_cast<size_t>(size));
  input.seekg(0);
  input.read(reinterpret_cast<char *>(bytes.data()), size);
  require(input.good(), "Cannot read input");
  return bytes;
}

static Bytes decode_atlas_png(const char *path) {
  const Bytes ktx = read_file(path);
  const std::array<uint8_t, 12> magic{0xab, 0x4b, 0x54, 0x58, 0x20, 0x31,
                                      0x31, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a};
  require(ktx.size() >= 68 &&
              std::equal(magic.begin(), magic.end(), ktx.begin()),
          "Expected KTX1 atlas");
  require(u32(ktx, 12) == 0x04030201 && u32(ktx, 16) == 0 &&
              u32(ktx, 24) == 0 && u32(ktx, 28) == 0x8e8c,
          "Expected little-endian BC7_UNORM KTX1");
  const uint32_t width = u32(ktx, 36), height = u32(ktx, 40);
  require(width && height && width <= 16384 && height <= 16384 &&
              width % 4 == 0 && height % 4 == 0,
          "Expected block-aligned 2D atlas dimensions");
  require(u32(ktx, 44) == 0 && u32(ktx, 48) == 0 && u32(ktx, 52) == 1 &&
              u32(ktx, 56) == 1,
          "Expected one-face, single-level 2D atlas");
  const size_t start = 64 + u32(ktx, 60);
  const size_t block_bytes = static_cast<size_t>(width) * height;
  require(u32(ktx, start) == block_bytes && start + 4 <= ktx.size() &&
              block_bytes == ktx.size() - start - 4,
          "Unexpected KTX1 atlas payload");
  Bytes rgba(static_cast<size_t>(width) * height * 4);
  for (size_t y = 0; y < height; y += 4)
    for (size_t x = 0; x < width; x += 4) {
      const size_t offset = start + 4 + (y / 4 * (width / 4) + x / 4) * 16;
      bcdec_bc7(ktx.data() + offset, rgba.data() + (y * width + x) * 4,
                static_cast<int>(width * 4));
    }
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  image.width = width;
  image.height = height;
  image.format = PNG_FORMAT_RGBA;
  png_alloc_size_t size = 0;
  require(png_image_write_to_memory(&image, nullptr, &size, 0, rgba.data(), 0,
                                    nullptr) != 0,
          "Cannot measure PNG");
  Bytes png(static_cast<size_t>(size));
  require(png_image_write_to_memory(&image, png.data(), &size, 0, rgba.data(),
                                    0, nullptr) != 0,
          "Cannot encode PNG");
  png.resize(static_cast<size_t>(size));
  return png;
}

static void append_u32(Bytes &bytes, uint32_t value) {
  const auto *first = reinterpret_cast<const uint8_t *>(&value);
  bytes.insert(bytes.end(), first, first + 4);
}

static void pack_generated(const char *mesh_path, const char *atlas_path,
                           const char *timings_path, const char *output_path,
                           const char *material_source) {
  require(!std::filesystem::exists(output_path), "Output already exists");
  std::ifstream timings_input(timings_path);
  require(timings_input.good(), "Cannot open baker normal-mode record");
  const Json timings = Json::parse(timings_input);
  const std::string normal_mode = timings.at("normal_mode");
  require(normal_mode == "implicit_flat" || normal_mode == "stored_vertex",
          "Unknown baker normal mode");
  auto mesh = painted_mesh::load(mesh_path);
  const size_t vertices = mesh.vertex_bytes.size() / 32;
  const size_t faces = mesh.triangles.size();
  Bytes binary = std::move(mesh.vertex_bytes);
  std::array<float, 3> minimum{}, maximum{};
  minimum.fill(std::numeric_limits<float>::infinity());
  maximum.fill(-std::numeric_limits<float>::infinity());
  for (size_t vertex = 0; vertex < vertices; ++vertex)
    for (size_t axis = 0; axis < 3; ++axis) {
      float value;
      std::memcpy(&value, binary.data() + vertex * 32 + axis * 4, 4);
      minimum[axis] = std::min(minimum[axis], value);
      maximum[axis] = std::max(maximum[axis], value);
    }
  const size_t index_start = binary.size();
  for (const auto &triangle : mesh.triangles)
    for (uint32_t index : triangle)
      append_u32(binary, index);
  const Bytes png = decode_atlas_png(atlas_path);
  const size_t image_start = binary.size();
  binary.insert(binary.end(), png.begin(), png.end());
  while (binary.size() % 4)
    binary.push_back(0);
  Json document = {
      {"asset", {{"version", "2.0"}}},
      {"scene", 0},
      {"scenes", {{{"nodes", {0}}}}},
      {"nodes", {{{"mesh", 0}}}},
      {"buffers", {{{"byteLength", binary.size()}}}},
      {"bufferViews",
       {{{"buffer", 0},
         {"byteOffset", 0},
         {"byteLength", vertices * 32},
         {"byteStride", 32},
         {"target", 34962}},
        {{"buffer", 0},
         {"byteOffset", index_start},
         {"byteLength", faces * 12},
         {"target", 34963}},
        {{"buffer", 0},
         {"byteOffset", image_start},
         {"byteLength", png.size()}}}},
      {"accessors",
       {{{"bufferView", 0},
         {"byteOffset", 0},
         {"componentType", 5126},
         {"count", vertices},
         {"type", "VEC3"},
         {"min", minimum},
         {"max", maximum}},
        {{"bufferView", 0},
         {"byteOffset", 12},
         {"componentType", 5126},
         {"count", vertices},
         {"type", "VEC3"}},
        {{"bufferView", 0},
         {"byteOffset", 24},
         {"componentType", 5126},
         {"count", vertices},
         {"type", "VEC2"}},
        {{"bufferView", 1},
         {"componentType", 5125},
         {"count", faces * 3},
         {"type", "SCALAR"}}}},
      {"images", {{{"bufferView", 2}, {"mimeType", "image/png"}}}},
      {"textures", {{{"source", 0}}}},
  };
  const Json material = {{"name", "painted_opaque"},
                         {"pbrMetallicRoughness",
                          {{"baseColorTexture", {{"index", 0}}},
                           {"metallicFactor", 0},
                           {"roughnessFactor", 1}}}};
  Json primitive = {
      {"attributes", {{"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}}},
      {"indices", 3},
      {"material", 0}};
  if (normal_mode == "implicit_flat") {
    // glTF specifies flat shading when NORMAL is absent. The baker's stored
    // helper vertex normals must not silently turn that into smooth shading.
    primitive["attributes"].erase("NORMAL");
    document["accessors"].erase(document["accessors"].begin() + 1);
    primitive["attributes"]["TEXCOORD_0"] = 1;
    primitive["indices"] = 2;
  }
  document["materials"] = Json::array({material});
  document["meshes"] =
      Json::array({{{"primitives", Json::array({primitive})}}});
  if (material_source) {
    const auto source_bytes = read_file(material_source);
    require(u32(source_bytes, 0) == 0x46546c67 && u32(source_bytes, 4) == 2,
            "Material source must be GLB 2");
    const size_t json_length = u32(source_bytes, 12);
    require(u32(source_bytes, 16) == 0x4e4f534a &&
                28 + json_length <= source_bytes.size(),
            "Invalid source GLB JSON");
    const auto source = Json::parse(source_bytes.begin() + 20,
                                    source_bytes.begin() + 20 + json_length);
    const size_t source_start = 28 + json_length;
    const size_t source_length = u32(source_bytes, 20 + json_length);
    require(u32(source_bytes, 24 + json_length) == 0x004e4942 &&
                source_length <= source_bytes.size() - source_start,
            "Invalid source GLB binary");
    require(source.at("materials").size() == 1,
            "Pack source materials into one UV layout before baking");
    const auto source_images = source.value("images", Json::array());
    const auto source_textures = source.value("textures", Json::array());
    const size_t missing = std::numeric_limits<size_t>::max();
    std::vector<size_t> image_indices(source_images.size(), missing);
    std::vector<size_t> texture_indices(source_textures.size(), missing);
    auto copy_image = [&](size_t index) {
      require(index < source_images.size(), "Source image index out of range");
      if (image_indices[index] != missing)
        return image_indices[index];
      auto image = source_images[index];
      require(image.contains("bufferView"), "Source images must be embedded");
      const auto &view =
          source.at("bufferViews").at(image["bufferView"].get<size_t>());
      require(view.at("buffer") == 0, "External source buffers unsupported");
      const size_t offset = view.value("byteOffset", size_t{0});
      const size_t length = view.at("byteLength");
      require(offset <= source_length && length <= source_length - offset,
              "Source image exceeds binary chunk");
      image["bufferView"] = document["bufferViews"].size();
      document["bufferViews"].push_back({{"buffer", 0},
                                         {"byteOffset", binary.size()},
                                         {"byteLength", length}});
      binary.insert(binary.end(), source_bytes.begin() + source_start + offset,
                    source_bytes.begin() + source_start + offset + length);
      while (binary.size() % 4)
        binary.push_back(0);
      const size_t destination = document["images"].size();
      document["images"].push_back(image);
      image_indices[index] = destination;
      return destination;
    };
    auto copy_texture = [&](size_t index) {
      require(index < source_textures.size(),
              "Source texture index out of range");
      if (texture_indices[index] != missing)
        return texture_indices[index];
      auto texture = source_textures[index];
      require(texture.contains("source"), "Source texture must name its image");
      texture["source"] = copy_image(texture["source"].get<size_t>());
      const size_t destination = document["textures"].size();
      document["textures"].push_back(texture);
      texture_indices[index] = destination;
      return destination;
    };
    if (source.contains("samplers"))
      document["samplers"] = source["samplers"];
    for (const auto *key : {"extensionsUsed", "extensionsRequired"})
      if (source.contains(key))
        document[key] = source[key];
    auto retained = source["materials"][0];
    // The new atlas replaces this image; copy only maps still used elsewhere.
    retained["pbrMetallicRoughness"].erase("baseColorTexture");
    remap_texture_references(retained, copy_texture);
    retained["pbrMetallicRoughness"]["baseColorTexture"] = {{"index", 0}};
    retained["pbrMetallicRoughness"]["baseColorFactor"] = {1, 1, 1, 1};
    document["materials"][0] = retained;
    document["buffers"][0]["byteLength"] = binary.size();
  }
  std::string json = document.dump();
  while (json.size() % 4)
    json.push_back(' ');
  require(binary.size() + json.size() + 28 <= UINT32_MAX,
          "GLB exceeds size limit");
  Bytes output;
  append_u32(output, 0x46546c67);
  append_u32(output, 2);
  append_u32(output, static_cast<uint32_t>(binary.size() + json.size() + 28));
  append_u32(output, static_cast<uint32_t>(json.size()));
  append_u32(output, 0x4e4f534a);
  output.insert(output.end(), json.begin(), json.end());
  append_u32(output, static_cast<uint32_t>(binary.size()));
  append_u32(output, 0x004e4942);
  output.insert(output.end(), binary.begin(), binary.end());
  std::ofstream file(output_path, std::ios::binary);
  file.write(reinterpret_cast<const char *>(output.data()),
             static_cast<std::streamsize>(output.size()));
  require(file.good(), "Cannot write packaged GLB");
  std::cout << "packed " << vertices << " vertices and " << faces
            << " triangles; UV/vertex bytes unchanged; BC7 decoded to PNG\n";
}

int main(int argc, char **argv) {
  static_assert(std::endian::native == std::endian::little);
  try {
    require(argc == 5 || argc == 6,
            "Usage: pack-generated sparse-uv.ply atlas-bc7.ktx "
            "bake-timings.json OUTPUT.glb [MATERIAL_SOURCE.glb]");
    pack_generated(argv[1], argv[2], argv[3], argv[4],
                   argc == 6 ? argv[5] : nullptr);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
