// CUDA/Vulkan external-memory boundary. Geometry never crosses host memory.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <png.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#define VK(call)                                                               \
  do {                                                                         \
    auto r = (call);                                                           \
    if (r != VK_SUCCESS)                                                       \
      throw std::runtime_error(#call + std::string(": ") + std::to_string(r)); \
  } while (0)
#define CU(call)                                                               \
  do {                                                                         \
    auto r = (call);                                                           \
    if (r != CUDA_SUCCESS) {                                                   \
      const char *s;                                                           \
      cuGetErrorString(r, &s);                                                 \
      throw std::runtime_error(#call + std::string(": ") + s);                 \
    }                                                                          \
  } while (0)
struct Buffer {
  VkBuffer buffer;
  VkDeviceMemory memory;
  VkDeviceSize size;
  CUexternalMemory external;
  CUdeviceptr cuda;
};
struct Image {
  VkImage image;
  VkImageView view;
  VkDeviceMemory memory;
  uint32_t width, height, layers;
};
struct Pipeline {
  VkPipeline pipeline;
  VkPipelineLayout layout;
  std::vector<VkDescriptorSetLayout> layouts;
  std::vector<VkDescriptorSet> sets;
};
struct Binding {
  uint32_t set, binding, type, id;
}; // types: 0 uniform, 1 storage, 2 sampled image, 3 sampler.
struct Context {
  VkInstance instance{};
  VkPhysicalDevice physical{};
  VkDevice device{};
  VkQueue queue{};
  uint32_t family{};
  VkCommandPool pool{};
  VkDescriptorPool descriptors{};
  VkSampler sampler{};
  VkPhysicalDeviceMemoryProperties memory_properties{};
  std::vector<Buffer> buffers;
  std::vector<Image> images;
  std::vector<Pipeline> pipelines;
  uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
      if ((bits & (1u << i)) &&
          (memory_properties.memoryTypes[i].propertyFlags & flags) == flags)
        return i;
    throw std::runtime_error("No device memory type");
  }
  Context() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Resident depth baker";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    VK(vkCreateInstance(&instance_info, nullptr, &instance));
    uint32_t count;
    VK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    VK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
    for (auto candidate : devices) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(candidate, &props);
      if (props.vendorID == 0x10de) {
        physical = candidate;
        break;
      }
    }
    if (!physical)
      throw std::runtime_error("NVIDIA Vulkan device unavailable");
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues.data());
    for (family = 0; family < count; family++)
      if (queues[family].queueFlags & VK_QUEUE_COMPUTE_BIT)
        break;
    float priority = 1;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueFamilyIndex = family;
    q.queueCount = 1;
    q.pQueuePriorities = &priority;
    const char *extensions[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
    VkPhysicalDeviceFeatures features{};
    features.textureCompressionBC = VK_TRUE;
    features.shaderInt64 = VK_TRUE;
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &q;
    info.enabledExtensionCount = 1;
    info.ppEnabledExtensionNames = extensions;
    info.pEnabledFeatures = &features;
    VK(vkCreateDevice(physical, &info, nullptr, &device));
    vkGetDeviceQueue(device, family, 0, &queue);
    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cp.queueFamilyIndex = family;
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK(vkCreateCommandPool(device, &cp, nullptr, &pool));
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 32},
                                    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64},
                                    {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 32},
                                    {VK_DESCRIPTOR_TYPE_SAMPLER, 16}};
    VkDescriptorPoolCreateInfo dp{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp.maxSets = 32;
    dp.poolSizeCount = 4;
    dp.pPoolSizes = sizes;
    VK(vkCreateDescriptorPool(device, &dp, nullptr, &descriptors));
    VkSamplerCreateInfo sp{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sp.magFilter = sp.minFilter = VK_FILTER_LINEAR;
    sp.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sp.addressModeU = sp.addressModeV = sp.addressModeW =
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sp.maxLod = 0;
    VK(vkCreateSampler(device, &sp, nullptr, &sampler));
  }
  ~Context() {
    vkDeviceWaitIdle(device);
    for (auto &p : pipelines) {
      vkDestroyPipeline(device, p.pipeline, nullptr);
      vkDestroyPipelineLayout(device, p.layout, nullptr);
      for (auto l : p.layouts)
        vkDestroyDescriptorSetLayout(device, l, nullptr);
    }
    for (auto &i : images) {
      vkDestroyImageView(device, i.view, nullptr);
      vkDestroyImage(device, i.image, nullptr);
      vkFreeMemory(device, i.memory, nullptr);
    }
    for (auto &b : buffers) {
      cuMemFree(b.cuda);
      cuDestroyExternalMemory(b.external);
      vkDestroyBuffer(device, b.buffer, nullptr);
      vkFreeMemory(device, b.memory, nullptr);
    }
    vkDestroySampler(device, sampler, nullptr);
    vkDestroyDescriptorPool(device, descriptors, nullptr);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
  }
  void ownership(VkCommandBuffer cmd, bool acquire) {
    std::vector<VkBufferMemoryBarrier> barriers;
    for (auto &b : buffers) {
      VkBufferMemoryBarrier x{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      x.buffer = b.buffer;
      x.size = VK_WHOLE_SIZE;
      x.srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_EXTERNAL : family;
      x.dstQueueFamilyIndex = acquire ? family : VK_QUEUE_FAMILY_EXTERNAL;
      x.srcAccessMask =
          acquire ? 0 : VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
      x.dstAccessMask =
          acquire ? VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT : 0;
      barriers.push_back(x);
    }
    if (!barriers.empty())
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                           barriers.size(), barriers.data(), 0, nullptr);
  }
  VkCommandBuffer begin() {
    CU(cuCtxSynchronize());
    VkCommandBufferAllocateInfo a{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    a.commandPool = pool;
    a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    a.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK(vkAllocateCommandBuffers(device, &a, &cmd));
    VkCommandBufferBeginInfo b{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    b.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK(vkBeginCommandBuffer(cmd, &b));
    ownership(cmd, true);
    return cmd;
  }
  void finish(VkCommandBuffer cmd) {
    ownership(cmd, false);
    VK(vkEndCommandBuffer(cmd));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    VK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(device, pool, 1, &cmd);
  }
  uint32_t buffer(size_t size) {
    Buffer b{};
    b.size = std::max(size, size_t(4));
    VkExternalMemoryBufferCreateInfo external{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.pNext = &external;
    info.size = b.size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VK(vkCreateBuffer(device, &info, nullptr, &b.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    VkExportMemoryAllocateInfo export_info{
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.pNext = &export_info;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex =
        memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK(vkAllocateMemory(device, &alloc, nullptr, &b.memory));
    VK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
    VkMemoryGetFdInfoKHR fd_info{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    fd_info.memory = b.memory;
    fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd;
    auto get_fd =
        (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    VK(get_fd(device, &fd_info, &fd));
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC handle{};
    handle.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    handle.handle.fd = fd;
    handle.size = req.size;
    CU(cuImportExternalMemory(&b.external, &handle));
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC mapping{};
    mapping.size = req.size;
    CU(cuExternalMemoryGetMappedBuffer(&b.cuda, b.external, &mapping));
    buffers.push_back(b);
    return buffers.size() - 1;
  }
  uint32_t image(uint32_t w, uint32_t h, uint32_t layers, VkFormat format) {
    Image i{};
    i.width = w;
    i.height = h;
    i.layers = layers;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {w, h, 1};
    info.mipLevels = 1;
    info.arrayLayers = layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VK(vkCreateImage(device, &info, nullptr, &i.image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, i.image, &req);
    VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    a.allocationSize = req.size;
    a.memoryTypeIndex =
        memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK(vkAllocateMemory(device, &a, nullptr, &i.memory));
    VK(vkBindImageMemory(device, i.image, i.memory, 0));
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = i.image;
    view.viewType =
        layers == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    view.format = format;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    VK(vkCreateImageView(device, &view, nullptr, &i.view));
    auto cmd = begin();
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.image = i.image;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex =
        VK_QUEUE_FAMILY_IGNORED;
    barrier.dstAccessMask =
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.subresourceRange = view.subresourceRange;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
    finish(cmd);
    images.push_back(i);
    return images.size() - 1;
  }
  uint32_t pipeline(const char *path, const Binding *bindings, uint32_t count,
                    const char *entry) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
      throw std::runtime_error("Missing SPIR-V shader");
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<uint32_t> code((size + 3) / 4);
    file.read((char *)code.data(), size);
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = size;
    mi.pCode = code.data();
    VkShaderModule module;
    VK(vkCreateShaderModule(device, &mi, nullptr, &module));
    Pipeline p{};
    uint32_t groups = 0;
    for (uint32_t k = 0; k < count; k++)
      groups = std::max(groups, bindings[k].set + 1);
    p.layouts.resize(groups);
    p.sets.resize(groups);
    const VkDescriptorType types[] = {
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER};
    for (uint32_t set = 0; set < groups; set++) {
      std::vector<VkDescriptorSetLayoutBinding> entries;
      for (uint32_t k = 0; k < count; k++)
        if (bindings[k].set == set)
          entries.push_back({bindings[k].binding, types[bindings[k].type], 1,
                             VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
      VkDescriptorSetLayoutCreateInfo li{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      li.bindingCount = entries.size();
      li.pBindings = entries.data();
      VK(vkCreateDescriptorSetLayout(device, &li, nullptr, &p.layouts[set]));
    }
    VkPipelineLayoutCreateInfo li{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = groups;
    li.pSetLayouts = p.layouts.data();
    VK(vkCreatePipelineLayout(device, &li, nullptr, &p.layout));
    VkComputePipelineCreateInfo ci{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.layout = p.layout;
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = entry;
    auto result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci,
                                           nullptr, &p.pipeline);
    vkDestroyShaderModule(device, module, nullptr);
    VK(result);
    VkDescriptorSetAllocateInfo da{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool = descriptors;
    da.descriptorSetCount = groups;
    da.pSetLayouts = p.layouts.data();
    VK(vkAllocateDescriptorSets(device, &da, p.sets.data()));
    for (uint32_t k = 0; k < count; k++) {
      auto b = bindings[k];
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.dstSet = p.sets[b.set];
      write.dstBinding = b.binding;
      write.descriptorCount = 1;
      write.descriptorType = types[b.type];
      VkDescriptorBufferInfo bi{};
      VkDescriptorImageInfo ii{};
      if (b.type < 2) {
        auto &r = buffers.at(b.id);
        bi = {r.buffer, 0, r.size};
        write.pBufferInfo = &bi;
      } else {
        if (b.type == 2) {
          ii.imageView = images.at(b.id).view;
          ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        } else
          ii.sampler = sampler;
        write.pImageInfo = &ii;
      }
      vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
    pipelines.push_back(p);
    return pipelines.size() - 1;
  }
};
static thread_local std::string error;
extern "C" {
const char *rd_error() { return error.c_str(); }
void *rd_create() {
  try {
    return new Context();
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
void rd_destroy(Context *c) { delete c; }
int rd_buffer(Context *c, size_t size) {
  try {
    return c->buffer(size);
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
int rd_write(Context *c, uint32_t id, uint64_t source, size_t size, int gpu) {
  try {
    auto &b = c->buffers.at(id);
    if (size > b.size)
      throw std::runtime_error("Buffer upload too large");
    if (gpu)
      CU(cuMemcpyDtoD(b.cuda, source, size));
    else
      CU(cuMemcpyHtoD(b.cuda, (void *)source, size));
    CU(cuCtxSynchronize());
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
int rd_read(Context *c, uint32_t id, void *target, size_t size) {
  try {
    auto &b = c->buffers.at(id);
    if (size > b.size)
      throw std::runtime_error("Buffer download too large");
    CU(cuMemcpyDtoH(target, b.cuda, size));
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
int rd_image(Context *c, uint32_t w, uint32_t h, uint32_t layers,
             uint32_t format) {
  try {
    return c->image(w, h, layers, (VkFormat)format);
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
int rd_copy_image(Context *c, uint32_t buffer, uint32_t image, uint32_t layer) {
  try {
    auto &i = c->images.at(image);
    auto cmd = c->begin();
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, layer, 1};
    copy.imageExtent = {i.width, i.height, 1};
    vkCmdCopyBufferToImage(cmd, c->buffers.at(buffer).buffer, i.image,
                           VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
    c->finish(cmd);
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
int rd_pipeline(Context *c, const char *path, const Binding *bindings,
                uint32_t count, const char *entry) {
  try {
    return c->pipeline(path, bindings, count, entry);
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
int rd_dispatch(Context *c, uint32_t id, uint32_t x, uint32_t y) {
  try {
    auto &p = c->pipelines.at(id);
    auto cmd = c->begin();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0,
                            p.sets.size(), p.sets.data(), 0, nullptr);
    vkCmdDispatch(cmd, x, y, 1);
    c->finish(cmd);
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
}
extern "C" int rd_read_at(Context *c, uint32_t id, size_t offset, void *target,
                          size_t size) {
  try {
    auto &b = c->buffers.at(id);
    if (offset > b.size || size > b.size - offset)
      throw std::runtime_error("Buffer read range");
    CU(cuMemcpyDtoH(target, b.cuda + offset, size));
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
extern "C" int rd_clear(Context *c, uint32_t id) {
  try {
    auto &b = c->buffers.at(id);
    CU(cuMemsetD8(b.cuda, 0, b.size));
    CU(cuCtxSynchronize());
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
// Final files are written at the output boundary, never used as stage handoffs.
extern "C" int rd_write_mesh(Context *c, uint32_t positions, uint32_t normals,
                             uint32_t uv, uint32_t indices, const char *path) {
  try {
    const auto vertices = c->buffers.at(positions).size / 12;
    const auto faces = c->buffers.at(indices).size / 12;
    std::vector<float> p(vertices * 3), n(vertices * 3), t(vertices * 2);
    std::vector<int32_t> f(faces * 3);
    CU(cuMemcpyDtoH(p.data(), c->buffers.at(positions).cuda, p.size() * 4));
    CU(cuMemcpyDtoH(n.data(), c->buffers.at(normals).cuda, n.size() * 4));
    CU(cuMemcpyDtoH(t.data(), c->buffers.at(uv).cuda, t.size() * 4));
    CU(cuMemcpyDtoH(f.data(), c->buffers.at(indices).cuda, f.size() * 4));
    std::ofstream output(path, std::ios::binary);
    output << "ply\nformat binary_little_endian 1.0\nelement vertex "
           << vertices << "\n";
    for (const char *name : {"x", "y", "z", "nx", "ny", "nz", "s", "t"})
      output << "property float " << name << "\n";
    output << "element face " << faces
           << "\nproperty list uchar int vertex_indices\nend_header\n";
    for (size_t i = 0; i < vertices; i++) {
      float row[] = {p[i * 3],     p[i * 3 + 1], p[i * 3 + 2], n[i * 3],
                     n[i * 3 + 1], n[i * 3 + 2], t[i * 2],     t[i * 2 + 1]};
      output.write((char *)row, sizeof(row));
    }
    for (size_t i = 0; i < faces; i++) {
      output.put(3);
      output.write((char *)(f.data() + i * 3), 12);
    }
    output.close();
    if (!output)
      throw std::runtime_error("Writing final mesh failed");
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
extern "C" int rd_write_atlas(Context *c, uint32_t id, const char *path) {
  try {
    auto &buffer = c->buffers.at(id);
    const uint8_t magic[] = {0xab, 0x4b, 0x54, 0x58, 0x20, 0x31,
                             0x31, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a};
    const uint32_t header[] = {0x04030201, 0,    1,    0,          0x8e8c,
                               0x1908,     4096, 4096, 0,          0,
                               1,          1,    0,    4096 * 4096};
    std::vector<char> blocks(buffer.size);
    CU(cuMemcpyDtoH(blocks.data(), buffer.cuda, buffer.size));
    std::ofstream output(path, std::ios::binary);
    output.write((char *)magic, sizeof(magic));
    output.write((char *)header, sizeof(header));
    output.write(blocks.data(), blocks.size());
    output.close();
    if (!output)
      throw std::runtime_error("Writing BC7 atlas failed");
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
// Decode stored samples directly into the upload buffer.
extern "C" int rd_decode_png(const char *path, void *destination,
                             uint32_t padded) {
  FILE *file = std::fopen(path, "rb");
  if (!file) {
    error = "Opening painting failed";
    return -1;
  }
  char message[256] = "Reading painting failed";
  auto on_error = [](png_structp png, png_const_charp text) {
    auto *target = static_cast<char *>(png_get_error_ptr(png));
    std::strncpy(target, text, 255);
    target[255] = 0;
    png_longjmp(png, 1);
  };
  png_structp png =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, message, on_error, nullptr);
  png_infop info = png ? png_create_info_struct(png) : nullptr;
  auto *rows =
      static_cast<png_bytep *>(std::malloc(size_t(padded) * sizeof(png_bytep)));
  if (!png || !info || !rows) {
    if (png)
      png_destroy_read_struct(&png, info ? &info : nullptr, nullptr);
    std::fclose(file);
    std::free(rows);
    error = "Creating PNG decoder failed";
    return -1;
  }
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(file);
    std::free(rows);
    error = message;
    return -1;
  }
  png_init_io(png, file);
  png_read_info(png, info);
  uint32_t width = png_get_image_width(png, info),
           height = png_get_image_height(png, info);
  if (width != height || width > padded)
    png_error(png, "Painting dimensions exceed upload buffer");
  int colour = png_get_color_type(png, info),
      bits = png_get_bit_depth(png, info);
  bool transparency = png_get_valid(png, info, PNG_INFO_tRNS) != 0;
  if (bits == 16)
    png_set_strip_16(png);
  if (colour == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png);
  if (colour == PNG_COLOR_TYPE_GRAY && bits < 8)
    png_set_expand_gray_1_2_4_to_8(png);
  if (transparency)
    png_set_tRNS_to_alpha(png);
  if (colour == PNG_COLOR_TYPE_GRAY || colour == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(png);
  if (!(colour & PNG_COLOR_MASK_ALPHA) && !transparency)
    png_set_add_alpha(png, 255, PNG_FILLER_AFTER);
  png_set_interlace_handling(png);
  png_read_update_info(png, info);
  for (uint32_t y = 0; y < height; y++)
    rows[y] = static_cast<png_bytep>(destination) +
              size_t(height - 1 - y) * padded * 4;
  png_read_image(png, rows);
  png_read_end(png, info);
  png_destroy_read_struct(&png, &info, nullptr);
  std::fclose(file);
  std::free(rows);
  return 0;
}
