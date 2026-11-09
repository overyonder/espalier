// Visible-face projection and material partitioning. GLB buffers are preserved;
// only index groups and material records are appended when applying a
// selection.
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <png.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using Json = nlohmann::json;
using Matrix = Eigen::Matrix4d;
using Point = Eigen::Vector4d;
using Bytes = std::vector<uint8_t>;
namespace fs = std::filesystem;
constexpr uint8_t unresolved = 255;
constexpr std::array<const char *, 5> class_names = {
    "opaque", "clear_glass", "pearlescent_glass", "metal", "emissive"};

static void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

static Bytes read_bytes(const fs::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(input.good(), "Cannot open " + path.string());
  const auto size = input.tellg();
  require(size >= 0, "Cannot measure " + path.string());
  Bytes bytes(static_cast<size_t>(size));
  input.seekg(0);
  input.read(reinterpret_cast<char *>(bytes.data()), size);
  require(input.good(), "Cannot read " + path.string());
  return bytes;
}

static void write_bytes(const fs::path &path, const Bytes &bytes) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  require(output.good(), "Cannot write " + path.string());
}

static Json read_json(const fs::path &path) {
  std::ifstream input(path);
  require(input.good(), "Cannot open " + path.string());
  return Json::parse(input);
}

static uint32_t read_u32(const Bytes &bytes, size_t offset) {
  require(offset <= bytes.size() && bytes.size() - offset >= 4,
          "Truncated binary field");
  uint32_t value;
  std::memcpy(&value, bytes.data() + offset, 4);
  return value;
}

static void append_u32(Bytes &bytes, uint32_t value) {
  const auto *start = reinterpret_cast<const uint8_t *>(&value);
  bytes.insert(bytes.end(), start, start + 4);
}

struct Primitive {
  size_t mesh;
  size_t primitive;
  size_t first_face;
  std::vector<Point> positions;
  std::vector<uint32_t> indices;
};

struct MeshFile {
  Json document;
  Bytes binary;
  Bytes source;
  std::vector<Primitive> primitives;
  size_t face_count = 0;
};

static size_t accessor_offset(const MeshFile &mesh, const Json &accessor,
                              size_t component_size, size_t components,
                              size_t &stride) {
  require(!accessor.contains("sparse"),
          "Sparse accessor requires expansion before classification");
  const auto &view = mesh.document.at("bufferViews")
                         .at(accessor.at("bufferView").get<size_t>());
  require(view.value("buffer", 0) == 0, "External GLB buffer is not supported");
  stride = view.value("byteStride", component_size * components);
  require(stride >= component_size * components, "Invalid accessor stride");
  const size_t local = accessor.value("byteOffset", size_t{0});
  const size_t count = accessor.at("count");
  const size_t length =
      count ? (count - 1) * stride + component_size * components : 0;
  require(local <= view.at("byteLength").get<size_t>() &&
              length <= view.at("byteLength").get<size_t>() - local,
          "Accessor exceeds its buffer view");
  const size_t offset = view.value("byteOffset", size_t{0}) + local;
  require(offset <= mesh.binary.size() && length <= mesh.binary.size() - offset,
          "Accessor exceeds GLB buffer");
  return offset;
}

static MeshFile load_mesh(const fs::path &path) {
  MeshFile result;
  result.source = read_bytes(path);
  require(read_u32(result.source, 0) == 0x46546c67 &&
              read_u32(result.source, 4) == 2,
          "Expected GLB version 2");
  require(read_u32(result.source, 8) == result.source.size(),
          "Incorrect GLB length");
  size_t offset = 12;
  bool json_seen = false, binary_seen = false;
  while (offset < result.source.size()) {
    const size_t length = read_u32(result.source, offset);
    const uint32_t kind = read_u32(result.source, offset + 4);
    offset += 8;
    require(length <= result.source.size() - offset, "Truncated GLB chunk");
    if (kind == 0x4e4f534a) {
      require(!json_seen, "Duplicate GLB JSON chunk");
      result.document = Json::parse(
          result.source.begin() + static_cast<ptrdiff_t>(offset),
          result.source.begin() + static_cast<ptrdiff_t>(offset + length));
      json_seen = true;
    } else if (kind == 0x004e4942) {
      require(!binary_seen, "Duplicate GLB BIN chunk");
      result.binary.assign(
          result.source.begin() + static_cast<ptrdiff_t>(offset),
          result.source.begin() + static_cast<ptrdiff_t>(offset + length));
      binary_seen = true;
    } else
      throw std::runtime_error(
          "Unknown GLB chunk cannot be preserved by this writer");
    offset += length;
  }
  require(json_seen && binary_seen, "GLB needs JSON and BIN chunks");
  // These generated objects are static single-instance meshes. Reject animated
  // or instanced inputs rather than assigning one selection to differing poses.
  require(!result.document.contains("skins") &&
              !result.document.contains("animations"),
          "Classify the static generated mesh before rigging");
  require(result.document.at("meshes").size() == 1,
          "Expected one generated mesh");
  size_t instances = 0;
  for (const auto &node : result.document.at("nodes")) {
    if (node.contains("mesh"))
      ++instances;
    require(!node.contains("matrix") && !node.contains("rotation") &&
                !node.contains("translation") && !node.contains("scale"),
            "Bake node transforms or supply the untransformed final mesh");
  }
  require(instances == 1, "Expected one generated mesh instance");
  const auto &primitives = result.document.at("meshes").at(0).at("primitives");
  for (size_t index = 0; index < primitives.size(); ++index) {
    const auto &primitive = primitives.at(index);
    require(primitive.value("mode", 4) == 4 && !primitive.contains("targets"),
            "Expected static triangle primitives");
    require(!primitive.contains("extensions"),
            "Decode primitive extensions before classification");
    Primitive decoded{0, index, result.face_count, {}, {}};
    const auto &positions =
        result.document.at("accessors")
            .at(primitive.at("attributes").at("POSITION").get<size_t>());
    require(positions.at("componentType") == 5126 &&
                positions.at("type") == "VEC3",
            "Expected float32 positions");
    size_t stride = 0;
    size_t start = accessor_offset(result, positions, 4, 3, stride);
    for (size_t vertex = 0; vertex < positions.at("count").get<size_t>();
         ++vertex) {
      std::array<float, 3> value{};
      std::memcpy(value.data(), result.binary.data() + start + vertex * stride,
                  12);
      require(std::all_of(value.begin(), value.end(),
                          [](float x) { return std::isfinite(x); }),
              "Non-finite vertex");
      decoded.positions.emplace_back(value[0], value[1], value[2], 1);
    }
    if (primitive.contains("indices")) {
      const auto &indices = result.document.at("accessors")
                                .at(primitive.at("indices").get<size_t>());
      const int type = indices.at("componentType");
      require(indices.at("type") == "SCALAR" &&
                  (type == 5121 || type == 5123 || type == 5125),
              "Invalid triangle indices");
      const size_t width = type == 5121 ? 1 : type == 5123 ? 2 : 4;
      start = accessor_offset(result, indices, width, 1, stride);
      for (size_t item = 0; item < indices.at("count").get<size_t>(); ++item) {
        uint32_t value = 0;
        std::memcpy(&value, result.binary.data() + start + item * stride,
                    width);
        require(value < decoded.positions.size(),
                "Triangle index exceeds positions");
        decoded.indices.push_back(value);
      }
    } else {
      for (size_t vertex = 0; vertex < decoded.positions.size(); ++vertex)
        decoded.indices.push_back(static_cast<uint32_t>(vertex));
    }
    require(decoded.indices.size() % 3 == 0, "Incomplete triangle");
    result.face_count += decoded.indices.size() / 3;
    result.primitives.push_back(std::move(decoded));
  }
  return result;
}

struct Image {
  uint32_t width = 0;
  uint32_t height = 0;
  Bytes pixels;
};

static Image read_mask(const fs::path &path) {
  png_image png{};
  png.version = PNG_IMAGE_VERSION;
  require(png_image_begin_read_from_file(&png, path.c_str()) != 0,
          "Cannot read mask " + path.string());
  png.format = PNG_FORMAT_GRAY;
  Image image{png.width, png.height, Bytes(PNG_IMAGE_SIZE(png))};
  const bool good = png_image_finish_read(&png, nullptr, image.pixels.data(), 0,
                                          nullptr) != 0;
  png_image_free(&png);
  require(good, "Cannot decode mask " + path.string());
  return image;
}

static Matrix read_matrix(const Json &values) {
  require(values.size() == 4, "Expected row-major 4x4 matrix");
  Matrix matrix;
  for (int row = 0; row < 4; ++row) {
    require(values.at(row).size() == 4, "Expected row-major 4x4 matrix");
    for (int column = 0; column < 4; ++column)
      matrix(row, column) = values.at(row).at(column).get<double>();
  }
  require(matrix.allFinite(), "Non-finite transform");
  return matrix;
}

static Image read_view_mask(const Json &view, const fs::path &base) {
  if (view.contains("mask"))
    return read_mask(base / view.at("mask").get<std::string>());
  Image result;
  require(!view.at("class_masks").empty(), "View has no semantic masks");
  for (const auto &entry : view.at("class_masks")) {
    const unsigned label = entry.at("class");
    require(label < 5, "Unknown semantic class");
    const Image mask = read_mask(base / entry.at("mask").get<std::string>());
    require(std::all_of(mask.pixels.begin(), mask.pixels.end(),
                        [](uint8_t x) { return x == 0 || x == 255; }),
            "Class masks must contain only 0 and 255");
    if (result.pixels.empty())
      result = {mask.width, mask.height, Bytes(mask.pixels.size(), unresolved)};
    require(result.width == mask.width && result.height == mask.height,
            "Class mask canvases differ");
    for (size_t pixel = 0; pixel < mask.pixels.size(); ++pixel) {
      if (!mask.pixels[pixel])
        continue;
      if (result.pixels[pixel] == unresolved)
        result.pixels[pixel] = static_cast<uint8_t>(label);
      else if (result.pixels[pixel] != label)
        result.pixels[pixel] = 254;
    }
  }
  for (auto &pixel : result.pixels)
    if (pixel == 254)
      pixel = unresolved;
  return result;
}

static std::vector<Point> clip_triangle(std::vector<Point> polygon) {
  for (int plane = 0; plane < 6; ++plane) {
    std::vector<Point> clipped;
    if (polygon.empty())
      break;
    const int axis = plane / 2;
    const double sign = plane % 2 ? -1 : 1;
    Point previous = polygon.back();
    double previous_distance = previous.w() + sign * previous[axis];
    for (const auto &current : polygon) {
      const double distance = current.w() + sign * current[axis];
      if ((distance >= 0) != (previous_distance >= 0)) {
        const double t = previous_distance / (previous_distance - distance);
        clipped.push_back(previous + t * (current - previous));
      }
      if (distance >= 0)
        clipped.push_back(current);
      previous = current;
      previous_distance = distance;
    }
    polygon = std::move(clipped);
  }
  return polygon;
}

static double edge(const Eigen::Vector3d &a, const Eigen::Vector3d &b, double x,
                   double y) {
  return (b.x() - a.x()) * (y - a.y()) - (b.y() - a.y()) * (x - a.x());
}

static std::vector<uint32_t> visible_faces(const MeshFile &mesh,
                                           const Matrix &transform,
                                           uint32_t width, uint32_t height) {
  const size_t size = static_cast<size_t>(width) * height;
  std::vector<uint32_t> face_ids(size, UINT32_MAX);
  std::vector<double> depths(size, std::numeric_limits<double>::infinity());
  for (const auto &primitive : mesh.primitives) {
    for (size_t triangle = 0; triangle < primitive.indices.size() / 3;
         ++triangle) {
      const auto clipped = clip_triangle(
          {transform * primitive.positions.at(primitive.indices[triangle * 3]),
           transform *
               primitive.positions.at(primitive.indices[triangle * 3 + 1]),
           transform *
               primitive.positions.at(primitive.indices[triangle * 3 + 2])});
      for (size_t fan = 1; fan + 1 < clipped.size(); ++fan) {
        std::array<Eigen::Vector3d, 3> points;
        const std::array<size_t, 3> corners{0, fan, fan + 1};
        for (size_t corner = 0; corner < 3; ++corner) {
          const auto &point = clipped[corners[corner]];
          points[corner] = {(.5 + .5 * point.x() / point.w()) * width,
                            (.5 - .5 * point.y() / point.w()) * height,
                            point.z() / point.w()};
        }
        const double area =
            edge(points[0], points[1], points[2].x(), points[2].y());
        if (std::abs(area) < 1e-12)
          continue;
        const int x0 =
            std::max(0, static_cast<int>(std::floor(std::min(
                            {points[0].x(), points[1].x(), points[2].x()}))));
        const int x1 =
            std::min(static_cast<int>(width) - 1,
                     static_cast<int>(std::ceil(std::max(
                         {points[0].x(), points[1].x(), points[2].x()}))));
        const int y0 =
            std::max(0, static_cast<int>(std::floor(std::min(
                            {points[0].y(), points[1].y(), points[2].y()}))));
        const int y1 =
            std::min(static_cast<int>(height) - 1,
                     static_cast<int>(std::ceil(std::max(
                         {points[0].y(), points[1].y(), points[2].y()}))));
        for (int y = y0; y <= y1; ++y)
          for (int x = x0; x <= x1; ++x) {
            const double a = edge(points[1], points[2], x + .5, y + .5) / area;
            const double b = edge(points[2], points[0], x + .5, y + .5) / area;
            const double c = 1 - a - b;
            if (std::min({a, b, c}) < 0)
              continue;
            const double depth =
                a * points[0].z() + b * points[1].z() + c * points[2].z();
            const size_t pixel =
                static_cast<size_t>(y) * width + static_cast<size_t>(x);
            if (depth < depths[pixel]) {
              depths[pixel] = depth;
              face_ids[pixel] =
                  static_cast<uint32_t>(primitive.first_face + triangle);
            }
          }
      }
    }
  }
  return face_ids;
}

static void classify_faces(const Json &job, const fs::path &base,
                           const MeshFile &mesh, const fs::path &output) {
  require(!fs::exists(output), "Selection output already exists");
  require(mesh.face_count < UINT32_MAX, "Too many faces");
  const Matrix mesh_to_render = read_matrix(job.at("mesh_to_render"));
  const auto cameras = read_json(base / job.at("cameras").get<std::string>());
  const auto &views = job.at("views");
  require(views.size() == cameras.at("cams").size(),
          "Each retained camera needs a matching mask");
  std::vector<std::array<uint32_t, 5>> votes(mesh.face_count);
  std::vector<uint32_t> observations(mesh.face_count, 0);
  std::vector<Matrix> transforms;
  std::vector<std::array<uint32_t, 2>> dimensions;
  for (size_t view = 0; view < views.size(); ++view) {
    const Image mask = read_view_mask(views.at(view), base);
    require(mask.width > 0 && mask.height > 0 && mask.width == mask.height,
            "Expected a square retained camera canvas");
    require(std::all_of(mask.pixels.begin(), mask.pixels.end(),
                        [](uint8_t x) { return x < 5 || x == unresolved; }),
            "Mask contains an unknown class ID");
    const auto &camera = cameras.at("cams").at(view);
    const Matrix world = read_matrix(camera.at("matrix"));
    require(world.fullPivLu().isInvertible(), "Singular camera transform");
    const double focal =
        2 * camera.at("lens").get<double>() / camera.at("sensor").get<double>();
    Matrix projection = Matrix::Zero();
    projection(0, 0) = projection(1, 1) = focal;
    projection(0, 2) = -2 * camera.at("shift").at(0).get<double>();
    projection(1, 2) = -2 * camera.at("shift").at(1).get<double>();
    projection(2, 2) = -(1000.0 + .01) / (1000.0 - .01);
    projection(2, 3) = -2 * 1000.0 * .01 / (1000.0 - .01);
    projection(3, 2) = -1;
    transforms.push_back(projection * world.inverse() * mesh_to_render);
    dimensions.push_back({mask.width, mask.height});
    const auto ids =
        visible_faces(mesh, transforms.back(), mask.width, mask.height);
    std::vector<std::array<uint32_t, 5>> counts(mesh.face_count);
    std::vector<uint32_t> visible(mesh.face_count, 0);
    for (size_t pixel = 0; pixel < ids.size(); ++pixel) {
      if (ids[pixel] == UINT32_MAX)
        continue;
      ++visible[ids[pixel]];
      if (mask.pixels[pixel] < 5)
        ++counts[ids[pixel]][mask.pixels[pixel]];
    }
    for (size_t face = 0; face < mesh.face_count; ++face) {
      if (visible[face] < job.at("minimum_pixels").get<uint32_t>())
        continue;
      ++observations[face];
      const auto winner =
          std::max_element(counts[face].begin(), counts[face].end());
      if (static_cast<double>(*winner) / visible[face] >=
          job.at("minimum_pixel_agreement").get<double>()) {
        ++votes[face][static_cast<size_t>(winner - counts[face].begin())];
      }
    }
    std::cout << "projected view " << view + 1 << '/' << views.size()
              << std::endl;
  }
  Bytes labels(mesh.face_count, unresolved);
  std::array<size_t, 6> totals{};
  for (size_t face = 0; face < labels.size(); ++face) {
    const auto winner =
        std::max_element(votes[face].begin(), votes[face].end());
    if (*winner >= job.at("minimum_views").get<uint32_t>() &&
        observations[face] &&
        static_cast<double>(*winner) / observations[face] >=
            job.at("minimum_view_agreement").get<double>()) {
      labels[face] = static_cast<uint8_t>(winner - votes[face].begin());
    }
  }
  if (job.contains("face_overrides")) {
    std::ifstream overrides(base / job.at("face_overrides").get<std::string>());
    require(overrides.good(), "Cannot open reviewed face overrides");
    size_t face;
    unsigned label;
    while (overrides >> face >> label) {
      require(face < labels.size() && label < 5,
              "Invalid reviewed face override");
      labels[face] = static_cast<uint8_t>(label);
    }
    require(overrides.eof(), "Malformed reviewed face override");
  }
  fs::create_directories(output);
  constexpr std::array<std::array<uint8_t, 3>, 6> colours{{{150, 150, 150},
                                                           {50, 190, 255},
                                                           {170, 100, 240},
                                                           {230, 170, 40},
                                                           {255, 245, 180},
                                                           {255, 30, 120}}};
  for (size_t view = 0; view < transforms.size(); ++view) {
    const auto [width, height] = dimensions[view];
    const auto ids = visible_faces(mesh, transforms[view], width, height);
    Bytes pixels(ids.size() * 3, 18);
    for (size_t pixel = 0; pixel < ids.size(); ++pixel) {
      if (ids[pixel] == UINT32_MAX)
        continue;
      const auto &colour =
          colours[labels[ids[pixel]] == unresolved ? 5 : labels[ids[pixel]]];
      std::copy(colour.begin(), colour.end(),
                pixels.begin() + static_cast<ptrdiff_t>(pixel * 3));
    }
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    image.width = width;
    image.height = height;
    image.format = PNG_FORMAT_RGB;
    const auto path = output / ("faces-view-" + std::to_string(view) + ".png");
    require(png_image_write_to_file(&image, path.c_str(), 0, pixels.data(), 0,
                                    nullptr) != 0,
            "Cannot write face preview");
  }
  write_bytes(output / "faces.u8", labels);
  // Exact source bytes bind the selection to its geometry, ordering and atlas.
  // This immutable copy is also the source for the later derivative GLB.
  write_bytes(output / "source.glb", mesh.source);
  std::ofstream decisions(output / "face-votes.tsv");
  decisions << "face\tclass\tvisible_views\topaque\tclear_glass\tpearlescent_"
               "glass\tmetal\temissive\n";
  for (size_t face = 0; face < labels.size(); ++face) {
    ++totals[labels[face] == unresolved ? 5 : labels[face]];
    decisions << face << '\t' << static_cast<unsigned>(labels[face]) << '\t'
              << observations[face];
    for (const auto count : votes[face])
      decisions << '\t' << count;
    decisions << '\n';
  }
  require(decisions.good(), "Cannot write face decisions");
  Json summary;
  for (size_t label = 0; label < 5; ++label)
    summary[class_names[label]] = totals[label];
  summary["unresolved"] = totals[5];
  std::ofstream(output / "counts.json") << summary.dump(2) << '\n';
  std::cout << summary.dump() << std::endl;
}

static void add_extension(Json &document, const char *name) {
  if (!document.contains("extensionsUsed"))
    document["extensionsUsed"] = Json::array();
  auto &extensions = document["extensionsUsed"];
  if (std::find(extensions.begin(), extensions.end(), name) == extensions.end())
    extensions.push_back(name);
}

static void apply_materials(const fs::path &selection, const fs::path &output,
                            bool reviewed) {
  require(reviewed,
          "Applying requires --reviewed after inspecting the face selection");
  require(!fs::exists(output), "Material output already exists");
  MeshFile mesh = load_mesh(selection / "source.glb");
  const auto original_binary = mesh.binary;
  const Bytes labels = read_bytes(selection / "faces.u8");
  require(labels.size() == mesh.face_count,
          "Selection does not match source face count");
  require(std::all_of(labels.begin(), labels.end(),
                      [](uint8_t x) { return x < 5; }),
          "Resolve unclassified faces before applying materials");
  Json result_primitives = Json::array();
  for (const auto &primitive : mesh.primitives) {
    const Json original = mesh.document.at("meshes")
                              .at(0)
                              .at("primitives")
                              .at(primitive.primitive);
    std::array<std::vector<uint32_t>, 3> groups;
    for (size_t face = 0; face < primitive.indices.size() / 3; ++face) {
      const uint8_t label = labels[primitive.first_face + face];
      const size_t group = label == 1 || label == 2 ? label : 0;
      groups[group].insert(
          groups[group].end(),
          primitive.indices.begin() + static_cast<ptrdiff_t>(face * 3),
          primitive.indices.begin() + static_cast<ptrdiff_t>(face * 3 + 3));
    }
    for (size_t group = 0; group < groups.size(); ++group) {
      if (groups[group].empty())
        continue;
      Json part = original;
      while (mesh.binary.size() % 4)
        mesh.binary.push_back(0);
      const size_t start = mesh.binary.size();
      for (const auto index : groups[group])
        append_u32(mesh.binary, index);
      const size_t view = mesh.document["bufferViews"].size();
      mesh.document["bufferViews"].push_back(
          {{"buffer", 0},
           {"byteOffset", start},
           {"byteLength", groups[group].size() * 4},
           {"target", 34963}});
      const size_t accessor = mesh.document["accessors"].size();
      mesh.document["accessors"].push_back({{"bufferView", view},
                                            {"componentType", 5125},
                                            {"count", groups[group].size()},
                                            {"type", "SCALAR"}});
      part["indices"] = accessor;
      if (group) {
        Json material = original.contains("material")
                            ? mesh.document.at("materials")
                                  .at(original.at("material").get<size_t>())
                            : Json::object();
        require(!material.value("extensions", Json::object())
                        .contains("KHR_materials_unlit") &&
                    !material.value("extensions", Json::object())
                         .contains("KHR_materials_pbrSpecularGlossiness"),
                "Transmission requires a metallic-roughness lit material");
        material["name"] = std::string("Advent_") + class_names[group];
        material["pbrMetallicRoughness"]["metallicFactor"] = 0;
        material["pbrMetallicRoughness"]["roughnessFactor"] = .06;
        material["extensions"]["KHR_materials_transmission"] = {
            {"transmissionFactor", 1.0}};
        material["extensions"]["KHR_materials_ior"] = {{"ior", 1.5}};
        // glTF transmission uses an opaque alpha surface; transmission
        // and alpha coverage are different. Unreal maps this to its
        // coloured-transmittance Substrate material on import.
        material["alphaMode"] = "OPAQUE";
        material.erase("alphaCutoff");
        part["material"] = mesh.document["materials"].size();
        mesh.document["materials"].push_back(material);
        add_extension(mesh.document, "KHR_materials_transmission");
        add_extension(mesh.document, "KHR_materials_ior");
      }
      result_primitives.push_back(std::move(part));
    }
  }
  mesh.document["meshes"][0]["primitives"] = result_primitives;
  mesh.document["buffers"][0]["byteLength"] = mesh.binary.size();
  std::string encoded = mesh.document.dump();
  while (encoded.size() % 4)
    encoded.push_back(' ');
  require(mesh.binary.size() + encoded.size() + 28 <= UINT32_MAX,
          "GLB exceeds format size limit");
  Bytes bytes;
  append_u32(bytes, 0x46546c67);
  append_u32(bytes, 2);
  append_u32(bytes,
             static_cast<uint32_t>(28 + encoded.size() + mesh.binary.size()));
  append_u32(bytes, static_cast<uint32_t>(encoded.size()));
  append_u32(bytes, 0x4e4f534a);
  bytes.insert(bytes.end(), encoded.begin(), encoded.end());
  append_u32(bytes, static_cast<uint32_t>(mesh.binary.size()));
  append_u32(bytes, 0x004e4942);
  bytes.insert(bytes.end(), mesh.binary.begin(), mesh.binary.end());
  write_bytes(output, bytes);
  const MeshFile verified = load_mesh(output);
  require(verified.face_count == mesh.face_count &&
              std::equal(original_binary.begin(), original_binary.end(),
                         verified.binary.begin()),
          "Geometry or atlas changed during material partitioning");
  std::cout << "preserved " << mesh.face_count << " faces and "
            << original_binary.size() << " original buffer bytes\n";
}

int main(int argc, char **argv) {
  static_assert(std::endian::native == std::endian::little);
  try {
    require(argc >= 4, "Usage: surface-regions classify JOB.json OUTPUT_DIR | "
                       "apply SELECTION_DIR OUTPUT.glb --reviewed");
    const std::string command = argv[1];
    if (command == "classify") {
      require(argc == 4, "classify needs JOB.json OUTPUT_DIR");
      const auto job = read_json(argv[2]);
      for (const char *threshold :
           {"minimum_pixel_agreement", "minimum_view_agreement"}) {
        const double value = job.at(threshold);
        require(value > .5 && value <= 1,
                "Agreement thresholds must exceed .5 and be at most 1");
      }
      require(job.at("minimum_views").get<int>() > 0 &&
                  job.at("minimum_pixels").get<int>() > 0,
              "Minimum views and pixels must be positive");
      const fs::path base = fs::absolute(argv[2]).parent_path();
      classify_faces(job, base,
                     load_mesh(base / job.at("mesh").get<std::string>()),
                     argv[3]);
    } else if (command == "apply") {
      require(argc == 5, "apply needs SELECTION_DIR OUTPUT.glb --reviewed");
      apply_materials(argv[2], argv[3], std::string(argv[4]) == "--reviewed");
    } else
      throw std::runtime_error("Unknown command " + command);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
