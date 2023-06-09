#include "execution.hpp"

#include <fstream>
#include <string_view>
#include <variant>

namespace cle
{

static auto
cudaDefines(const ParameterMap & parameter_list, const ConstantMap & constant_list) -> std::string
{
  // @CherifMZ TODO: write cuda Defines to transform ocl Kernel into compatible cuda kernel
  // See
  // https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/pyclesperanto_prototype/_tier0/_cuda_execute.py

  std::ostringstream defines;
  defines << "\n#define get_global_size(dim) global_size_ ## dim ## _size";
  defines << "\n";

  defines << "\n#define GET_IMAGE_WIDTH(image_key) IMAGE_SIZE_ ## image_key ## "
             "_WIDTH";
  defines << "\n#define GET_IMAGE_HEIGHT(image_key) IMAGE_SIZE_ ## image_key "
             "## _HEIGHT";
  defines << "\n#define GET_IMAGE_DEPTH(image_key) IMAGE_SIZE_ ## image_key ## "
             "_DEPTH";
  defines << "\n";

  if (!constant_list.empty())
  {
    for (const auto & [key, value] : constant_list)
    {
      defines << "#define " << key << " " << value << "\n";
    }
    defines << "\n";
  }

  std::string size_params = "int global_size_0_size, int global_size_1_size, "
                            "int global_size_2_size, ";

  for (const auto & param : parameter_list)
  {
    if (std::holds_alternative<float>(param.second) || std::holds_alternative<int>(param.second))
    {
      continue;
    }
    const auto & arr = std::get<Array>(param.second);

    std::string ndim;
    std::string pos_type;
    std::string pos;

    std::string pixel_type;
    std::string type_id;

    if (arr.dim() < 3)
    {
      ndim = "2";
      pos_type = "int2";
      pos = "(pos0, pos1)";
    }
    else
    {
      ndim = "3";
      pos_type = "int4";
      pos = "(pos0, pos1, pos2, 0)";
    }

    std::string width = "image_" + param.first + "_width";
    std::string height = "image_" + param.first + "_height";
    std::string depth = "image_" + param.first + "_depth";

    size_params = size_params + "int " + width + ", int " + height + ", int " + depth + ", ";

    defines << "\n";
    defines << "\n#define IMAGE_SIZE_" << param.first << "_WIDTH " << width;
    defines << "\n#define IMAGE_SIZE_" << param.first << "_HEIGHT " << height;
    defines << "\n#define IMAGE_SIZE_" << param.first << "_DEPTH " << depth;
    defines << "\n";

    defines << "\n";
    defines << "\n#define CONVERT_" << param.first << "_PIXEL_TYPE clij_convert_" << arr.dtype() << "_sat";
    defines << "\n#define IMAGE_" << param.first << "_PIXEL_TYPE " << arr.dtype() << "";
    defines << "\n#define POS_" << param.first << "_TYPE " << pos_type;
    defines << "\n#define POS_" << param.first << "_INSTANCE(pos0,pos1,pos2,pos3) make_" << pos_type << "" << pos;
    defines << "\n";

    defines << "\n";
    defines << "\n#define IMAGE_" << param.first << "_TYPE " << size_params << "" << arr.dtype() << "*";
    defines << "\n#define READ_" << param.first << "_IMAGE(a,b,c) read_buffer" << ndim << "d" << arr.shortType()
            << "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)";
    defines << "\n#define WRITE_" << param.first << "_IMAGE(a,b,c) write_buffer" << ndim << "d" << arr.shortType()
            << "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)";
    defines << "\n";
  }

  defines << "\n";
  return defines.str();
}

static auto
oclDefines(const ParameterMap & parameter_list, const ConstantMap & constant_list) -> std::string
{
  std::ostringstream defines;
  defines << "\n#define GET_IMAGE_WIDTH(image_key) IMAGE_SIZE_ ## image_key ## _WIDTH";
  defines << "\n#define GET_IMAGE_HEIGHT(image_key) IMAGE_SIZE_ ## image_key ## _HEIGHT";
  defines << "\n#define GET_IMAGE_DEPTH(image_key) IMAGE_SIZE_ ## image_key ## _DEPTH";
  defines << "\n";

  for (const auto & param : parameter_list)
  {
    if (std::holds_alternative<float>(param.second) || std::holds_alternative<int>(param.second))
    {
      continue;
    }
    const auto & arr = std::get<Array>(param.second);

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
    defines << "\n#define CONVERT_" << param.first << "_PIXEL_TYPE clij_convert_" << arr.dtype() << "_sat";
    defines << "\n#define IMAGE_" << param.first << "_PIXEL_TYPE " << arr.dtype() << "";
    defines << "\n#define POS_" << param.first << "_TYPE " << pos_type;
    defines << "\n#define POS_" << param.first << "_INSTANCE(pos0,pos1,pos2,pos3) (" << pos_type << ")" << pos;
    defines << "\n";

    // define specific information
    if (true) // @StRigaud TODO: introduce cl_image / cudaArray
    {
      defines << "\n#define IMAGE_" << param.first << "_TYPE __global " << arr.dtype() << "*";
      defines << "\n#define READ_" << param.first << "_IMAGE(a,b,c) read_buffer" << ndim << "d" << arr.shortType()
              << "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)";
      defines << "\n#define WRITE_" << param.first << "_IMAGE(a,b,c) write_buffer" << ndim << "d" << arr.shortType()
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
      defines << "\n#define IMAGE_" << param.first << "_TYPE " << img_type_name;
      defines << "\n#define READ_" << param.first << "_IMAGE(a,b,c) read_image" << prefix << "(a,b,c)";
      defines << "\n#define WRITE_" << param.first << "_IMAGE(a,b,c) write_image" << prefix << "(a,b,c)";
    }

    // define size information
    defines << "\n";
    defines << "\n#define IMAGE_SIZE_" << param.first << "_WIDTH " << std::to_string(arr.width());
    defines << "\n#define IMAGE_SIZE_" << param.first << "_HEIGHT " << std::to_string(arr.height());
    defines << "\n#define IMAGE_SIZE_" << param.first << "_DEPTH " << std::to_string(arr.depth());
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

auto
execute(const DevicePtr &    device,
        const KernelInfo &   kernel_func,
        const ParameterMap & parameters,
        const ConstantMap &  constants,
        const RangeArray &   global_range) -> void
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

  // getPreamble: not implemented yet
  std::string program_source;
  std::string preamble = cle::BackendManager::getInstance().getBackend().getPreamble();
  std::string kernel_name = kernel_func.first;
  std::string kernel_source = kernel_func.second;
  program_source.reserve(preamble.size() + defines.size() + kernel_source.size());
  program_source += defines;
  program_source += preamble;
  program_source += kernel_source;

  // list kernel arguments and sizes
  for (const auto & param : parameters)
  {
    if (std::holds_alternative<Array>(param.second))
    {
      const auto & arr = std::get<Array>(param.second);
      args_ptr.push_back(*arr.get());
      if (device->getType() == Device::Type::CUDA)
      {
        args_size.push_back(arr.nbElements() * arr.bytesPerElements());
      }
      else if (device->getType() == Device::Type::OPENCL)
      {
        args_size.push_back(sizeof(cl_mem));
      }
      std::cout << "parameter " << param.first << " - " << std::get<Array>(param.second) << std::endl;
    }
    else if (std::holds_alternative<float>(param.second))
    {
      const auto & f = std::get<float>(param.second);
      args_ptr.push_back(const_cast<float *>(&f));
      args_size.push_back(sizeof(float));
      std::cout << "parameter " << param.first << " - " << std::get<float>(param.second) << std::endl;
    }
    else if (std::holds_alternative<int>(param.second))
    {
      const auto & i = std::get<int>(param.second);
      args_ptr.push_back(const_cast<int *>(&i));
      args_size.push_back(sizeof(int));
      std::cout << "parameter " << param.first << " - " << std::get<int>(param.second) << std::endl;
    }
  }

  // @CherifMZ: for TEST only, I added this chunk of code to write the content of the source into an external file
  std::ofstream file("kernel_output.txt");
  // Check if the file was opened successfully
  if (file.is_open())
  {
    // Write the content to the file
    file << program_source;
    // Close the file
    file.close();
  }

  // @StRigaud TODO: save source into file for debugging
  // @StRigaud TODO: call execution based on backend, warning dealing with void** and void* is not safe
  try
  {
    std::cout << "Execute kernel: " << kernel_name << std::endl;
    cle::BackendManager::getInstance().getBackend().executeKernel(
      device, program_source, kernel_name, global_range, args_ptr, args_size);
  }
  catch (const std::exception & e)
  {
    throw std::runtime_error("Error while executing kernel : " + std::string(e.what()));
  }
}

} // namespace cle
