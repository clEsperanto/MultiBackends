#include "execution.hpp"
#include "backend.hpp"

#include <fstream>
#include <string_view>

namespace cle
{

// Helper function for word replacements
static auto
replaceWord(std::string & sentence, const std::string_view & wordToReplace, const std::string_view & replacement)
  -> void
{
  size_t pos = sentence.find(wordToReplace);
  while (pos != std::string::npos)
  {
    sentence.replace(pos, wordToReplace.length(), replacement);
    pos = sentence.find(wordToReplace, pos + replacement.length());
  }
}

// Helper function for OpenCL to Cuda translation
static auto
srcOpenclToCuda(std::string opencl_code) -> std::string
{
  replaceWord(opencl_code, "(int2){", "make_int2(");
  replaceWord(opencl_code, "(int4){", "make_int4(");
  replaceWord(opencl_code, "(int4)  {", "make_int4(");
  replaceWord(opencl_code, "(float4){", "make_float4(");
  replaceWord(opencl_code, "(float2){", "make_float2(");
  replaceWord(opencl_code, "int2 pos = {", "int2 pos = make_int2(");
  replaceWord(opencl_code, "int4 pos = {", "int4 pos = make_int4(");
  replaceWord(opencl_code, "};", ");");
  replaceWord(opencl_code, "})", "))");

  replaceWord(opencl_code, "(int2)", "make_int2");
  replaceWord(opencl_code, "(int4)", "make_int4");
  replaceWord(opencl_code, "__constant sampler_t", "__device__ int");
  replaceWord(opencl_code, "__const sampler_t", "__device__ int");
  replaceWord(opencl_code, "inline", "__device__ inline");
  replaceWord(opencl_code, "#pragma", "// #pragma");

  replaceWord(opencl_code, "\nkernel void", "\nextern \"C\" __global__ void");
  replaceWord(opencl_code, "__kernel ", "extern \"C\" __global__ ");

  replaceWord(opencl_code, "get_global_id(0)", "blockDim.x * blockIdx.x + threadIdx.x");
  replaceWord(opencl_code, "get_global_id(1)", "blockDim.y * blockIdx.y + threadIdx.y");
  replaceWord(opencl_code, "get_global_id(2)", "blockDim.z * blockIdx.z + threadIdx.z");

  return opencl_code;
}

static auto
cudaDefines(const ParameterList & parameter_list, const ConstantList & constant_list) -> std::string
{
  // @CherifMZ TODO: write cuda Defines to transform ocl Kernel into compatible cuda kernel

  std::ostringstream defines;

  if (!constant_list.empty())
  {
    for (const auto & [key, value] : constant_list)
    {
      defines << "#define " << key << " " << value << "\n";
    }
    defines << "\n";
  }

  std::string size_params = "";
  for (const auto & param : parameter_list)
  {
    if (std::holds_alternative<const float>(param.second) || std::holds_alternative<const int>(param.second))
    {
      continue;
    }
    const auto & arr = std::get<Array::Pointer>(param.second);

    std::string ndim;
    std::string pos_type;
    std::string pos;
    std::string pixel_type;
    std::string type_id;
    switch (arr->dim())
    {
      case 1:
        ndim = "1";
        pos_type = "int";
        pos = "(pos0)";
        defines << "\n#define POS_" << param.first << "_INSTANCE(pos0,pos1,pos2,pos3) " << pos;
        break;
      case 2:
        ndim = "2";
        pos_type = "int2";
        pos = "(pos0, pos1)";
        defines << "\n#define POS_" << param.first << "_INSTANCE(pos0,pos1,pos2,pos3) make_" << pos_type << "" << pos;
        break;
      case 3:
      default:
        ndim = "3";
        pos_type = "int4";
        pos = "(pos0, pos1, pos2, 0)";
        defines << "\n#define POS_" << param.first << "_INSTANCE(pos0,pos1,pos2,pos3) make_" << pos_type << "" << pos;
        break;
    }

    defines << "\n";
    defines << "\n#define CONVERT_" << param.first << "_PIXEL_TYPE clij_convert_" << arr->dtype() << "_sat";
    defines << "\n#define IMAGE_" << param.first << "_PIXEL_TYPE " << arr->dtype() << "";
    defines << "\n#define POS_" << param.first << "_TYPE " << pos_type;
    defines << "\n";

    defines << "\n";
    defines << "\n#define IMAGE_SIZE_" << param.first << "_WIDTH " << std::to_string(arr->width());
    defines << "\n#define IMAGE_SIZE_" << param.first << "_HEIGHT " << std::to_string(arr->height());
    defines << "\n#define IMAGE_SIZE_" << param.first << "_DEPTH " << std::to_string(arr->depth());
    defines << "\n";

    defines << "\n";
    defines << "\n#define IMAGE_" << param.first << "_TYPE " << size_params << "" << arr->dtype() << "*";
    defines << "\n#define READ_" << param.first << "_IMAGE(a,b,c) read_buffer" << ndim << "d" << arr->shortType()
            << "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)";
    defines << "\n#define WRITE_" << param.first << "_IMAGE(a,b,c) write_buffer" << ndim << "d" << arr->shortType()
            << "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)";
    defines << "\n";

    size_params = "";
  }
  defines << "\n";

  return defines.str();
}

static auto
oclDefines(const ParameterList & parameter_list, const ConstantList & constant_list) -> std::string
{
  std::ostringstream defines;

  if (!constant_list.empty())
  {
    for (const auto & [key, value] : constant_list)
    {
      defines << "#define " << key << " " << value << "\n";
    }
    defines << "\n";
  }

  defines << "\n#define GET_IMAGE_WIDTH(image_key) IMAGE_SIZE_ ## image_key ## _WIDTH";
  defines << "\n#define GET_IMAGE_HEIGHT(image_key) IMAGE_SIZE_ ## image_key ## _HEIGHT";
  defines << "\n#define GET_IMAGE_DEPTH(image_key) IMAGE_SIZE_ ## image_key ## _DEPTH";
  defines << "\n";

  for (const auto & param : parameter_list)
  {
    if (std::holds_alternative<const float>(param.second) || std::holds_alternative<const int>(param.second))
    {
      continue;
    }
    const auto & arr = std::get<Array::Pointer>(param.second);

    std::string pos_type;
    std::string pos;
    std::string ndim;
    switch (arr->dim())
    {
      case 1:
        ndim = "1";
        pos_type = "int";
        pos = "(pos0)";
        break;
      case 2:
        ndim = "2";
        pos_type = "int2";
        pos = "(pos0, pos1)";
        break;
      case 3:
        ndim = "3";
        pos_type = "int4";
        pos = "(pos0, pos1, pos2, 0)";
        break;
      default:
        ndim = "3";
        pos_type = "int4";
        pos = "(pos0, pos1, pos2, 0)";
        break;
    }

    defines << "\n";
    defines << "\n#define CONVERT_" << param.first << "_PIXEL_TYPE clij_convert_" << arr->dtype() << "_sat";
    defines << "\n#define IMAGE_" << param.first << "_PIXEL_TYPE " << arr->dtype() << "";
    defines << "\n#define POS_" << param.first << "_TYPE " << pos_type;
    defines << "\n#define POS_" << param.first << "_INSTANCE(pos0,pos1,pos2,pos3) (" << pos_type << ")" << pos;
    defines << "\n";

    if (arr->mtype() == mType::BUFFER) // @StRigaud TODO: introduce cl_image / cudaArray
    {
      defines << "\n#define IMAGE_" << param.first << "_TYPE __global " << arr->dtype() << "*";
      defines << "\n#define READ_" << param.first << "_IMAGE(a,b,c) read_buffer" << ndim << "d" << arr->shortType()
              << "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)";
      defines << "\n#define WRITE_" << param.first << "_IMAGE(a,b,c) write_buffer" << ndim << "d" << arr->shortType()
              << "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)";
    }
    else
    {
      std::string img_type_name;
      if (param.first.find("dst") != std::string::npos || param.first.find("destination") != std::string::npos ||
          param.first.find("output") != std::string::npos)
      {
        img_type_name = "__write_only image" + ndim + "d_t";
      }
      else
      {
        img_type_name = "__read_only image" + ndim + "d_t";
      }
      std::string prefix;
      switch (arr->shortType().front())
      {
        case 'u':
          prefix = "ui";
          break;
        case 'f':
          prefix = "f";
          break;
        default:
          prefix = "i";
          break;
      }
      defines << "\n#define IMAGE_" << param.first << "_TYPE " << img_type_name;
      defines << "\n#define READ_" << param.first << "_IMAGE(a,b,c) read_image" << prefix << "(a,b,c)";
      defines << "\n#define WRITE_" << param.first << "_IMAGE(a,b,c) write_image" << prefix << "(a,b,c)";
    }

    defines << "\n";
    defines << "\n#define IMAGE_SIZE_" << param.first << "_WIDTH " << std::to_string(arr->width());
    defines << "\n#define IMAGE_SIZE_" << param.first << "_HEIGHT " << std::to_string(arr->height());
    defines << "\n#define IMAGE_SIZE_" << param.first << "_DEPTH " << std::to_string(arr->depth());
    defines << "\n";
  }
  defines << "\n";

  return defines.str();
}

auto
execute(const Device::Pointer & device,
        const KernelInfo &      kernel_func,
        const ParameterList &   parameters,
        const ConstantList &    constants,
        const RangeArray &      global_range) -> void
{
  // build program source
  std::string program_source;
  std::string preamble = cle::BackendManager::getInstance().getBackend().getPreamble();
  std::string kernel_name = kernel_func.first;
  std::string kernel_source = kernel_func.second;
  std::string defines;
  switch (device->getType())
  {
    case Device::Type::CUDA:
      defines = cle::cudaDefines(parameters, constants);
      kernel_source = cle::srcOpenclToCuda(kernel_source);
      break;
    case Device::Type::OPENCL:
      defines = cle::oclDefines(parameters, constants);
      break;
  }
  program_source.reserve(preamble.size() + defines.size() + kernel_source.size());
  program_source += defines;
  program_source += preamble;
  program_source += kernel_source;

  // prepare parameters to be passed to the backend
  std::vector<void *> args_ptr;
  std::vector<size_t> args_size;
  args_ptr.reserve(parameters.size());
  args_size.reserve(parameters.size());
  for (const auto & param : parameters)
  {
    if (std::holds_alternative<Array::Pointer>(param.second))
    {
      const auto & arr = std::get<Array::Pointer>(param.second);
      switch (device->getType())
      {
        case Device::Type::CUDA:
          args_ptr.push_back(arr->get());
          break;
        case Device::Type::OPENCL:
          args_ptr.push_back(*arr->get());
          args_size.push_back(sizeof(cl_mem));
          break;
      }
    }
    else if (std::holds_alternative<const float>(param.second))
    {
      const auto & f = std::get<const float>(param.second);
      args_ptr.push_back(const_cast<float *>(&f));
      args_size.push_back(sizeof(float));
    }
    else if (std::holds_alternative<const int>(param.second))
    {
      const auto & i = std::get<const int>(param.second);
      args_ptr.push_back(const_cast<int *>(&i));
      args_size.push_back(sizeof(int));
    }
  }

  // execute kernel
  try
  {
    cle::BackendManager::getInstance().getBackend().executeKernel(
      device, program_source, kernel_name, global_range, args_ptr, args_size);
  }
  catch (const std::exception & e)
  {
    throw std::runtime_error("Error: Failed to execute the kernel. \n\t > " + std::string(e.what()));
  }
}

} // namespace cle
