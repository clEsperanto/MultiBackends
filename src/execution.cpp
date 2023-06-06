
#include "execution.hpp"

namespace cle
{

static auto
cudaDefines(const ParameterMap & parameter_list, const ConstantMap & constant_list) -> std::string
{
  // @StRigaud TODO: write cuda Defines to transform ocl Kernel into compatible cuda kernel
  // See
  // https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/pyclesperanto_prototype/_tier0/_cuda_execute.py
  return "";
}

static auto
oclDefines(const ParameterMap & parameter_list, const ConstantMap & constant_list) -> std::string
{
  std::ostringstream defines;
  defines << "\n#define GET_IMAGE_WIDTH(image_key) IMAGE_SIZE_ ## image_key ## _WIDTH";
  defines << "\n#define GET_IMAGE_HEIGHT(image_key) IMAGE_SIZE_ ## image_key ## _HEIGHT";
  defines << "\n#define GET_IMAGE_DEPTH(image_key) IMAGE_SIZE_ ## image_key ## _DEPTH";
  defines << "\n";

  for (const auto & [key, value] : parameter_list)
  {
    if (std::holds_alternative<float>(value) || std::holds_alternative<int>(value))
    {
      continue;
    }
    const auto & arr = std::get<Array>(value);

    // manage dimensions and coordinates
    std::string pos_type;
    std::string pos;
    std::string ndim;
    switch (arr.dim())
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

    // define common information
    defines << "\n";
    defines << "\n#define CONVERT_" << key << "_PIXEL_TYPE clij_convert_" << arr.dtype() << "_sat";
    defines << "\n#define IMAGE_" << key << "_PIXEL_TYPE " << arr.dtype() << "";
    defines << "\n#define POS_" << key << "_TYPE " << pos_type;
    defines << "\n#define POS_" << key << "_INSTANCE(pos0,pos1,pos2,pos3) (" << pos_type << ")" << pos;
    defines << "\n";

    // define specific information
    if (true) // @StRigaud TODO: introduce cl_image / cudaArray
    {
      defines << "\n#define IMAGE_" << key << "_TYPE __global " << arr.dtype() << "*";
      defines << "\n#define READ_" << key << "_IMAGE(a,b,c) read_buffer" << ndim << "d" << arr.shortType()
              << "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)";
      defines << "\n#define WRITE_" << key << "_IMAGE(a,b,c) write_buffer" << ndim << "d" << arr.shortType()
              << "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)";
    }
    else
    {
      std::string img_type_name;
      if (key.find("dst") != std::string::npos || key.find("destination") != std::string::npos ||
          key.find("output") != std::string::npos)
      {
        img_type_name = "__write_only image" + ndim + "d_t";
      }
      else
      {
        img_type_name = "__read_only image" + ndim + "d_t";
      }
      std::string prefix;
      switch (arr.shortType().front())
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
      defines << "\n#define IMAGE_" << key << "_TYPE " << img_type_name;
      defines << "\n#define READ_" << key << "_IMAGE(a,b,c) read_image" << prefix << "(a,b,c)";
      defines << "\n#define WRITE_" << key << "_IMAGE(a,b,c) write_image" << prefix << "(a,b,c)";
    }

    // define size information
    defines << "\n";
    defines << "\n#define IMAGE_SIZE_" << key << "_WIDTH " << std::to_string(arr.width());
    defines << "\n#define IMAGE_SIZE_" << key << "_HEIGHT " << std::to_string(arr.height());
    defines << "\n#define IMAGE_SIZE_" << key << "_DEPTH " << std::to_string(arr.depth());
    defines << "\n";
  }

  // add constant memory defines
  if (!constant_list.empty())
  {
    for (const auto & [key, value] : constant_list)
    {
      defines << "#define " << key << " " << value << "\n";
    }
    defines << "\n";
  }

  // return defines as string
  defines << "\n";
  return defines.str();
}

static auto
execute(const DevicePtr &    device,
        const KernelInfo &   kernel_func,
        const ParameterMap & parameters,
        const ConstantMap &  constants,
        const RangeArray &   global_rage) -> void
{
  std::vector<void *> args_ptr;
  std::vector<size_t> args_size;
  args_ptr.reserve(parameters.size());
  args_size.reserve(parameters.size());

  // build kernel source
  std::string defines;
  switch (device->getType())
  {
    case Device::Type::CUDA:
      defines = cle::cudaDefines(parameters, constants);
      break;
    case Device::Type::OPENCL:
      defines = cle::oclDefines(parameters, constants);
      break;
  }
  std::string preamble = cle::BackendManager::getInstance().getBackend().getPreamble();
  std::string kernel = kernel_func.second;
  std::string func_name = kernel_func.first;
  std::string source = defines + preamble + kernel;

  // list kernel arguments and sizes
  for (const auto & [key, value] : parameters)
  {
    if (std::holds_alternative<Array>(value))
    {
      const auto & arr = std::get<Array>(value);
      args_ptr.push_back(*arr.get());
      args_size.push_back(arr.nbElements() * arr.bytesPerElements());
    }
    else if (std::holds_alternative<float>(value))
    {
      const auto & f = std::get<float>(value);
      args_ptr.push_back(const_cast<float *>(&f));
      args_size.push_back(sizeof(float));
    }
    else if (std::holds_alternative<int>(value))
    {
      const auto & i = std::get<int>(value);
      args_ptr.push_back(const_cast<int *>(&i));
      args_size.push_back(sizeof(int));
    }
  }

  // @StRigaud TODO: save source into file for debugging
  // @StRigaud TODO: call execution based on backend, warning dealing with void** and void* is not safe
  cle::BackendManager::getInstance().getBackend().executeKernel(
    device, source, func_name, global_rage, args_ptr, args_size);
}

} // namespace cle