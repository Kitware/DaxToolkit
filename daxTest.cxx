/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "dAPI.cl.h"


#include "daxArray.h"
#include "daxCellAverageModule.h"
#include "daxCellGradientModule.h"
#include "daxElevationModule.h"
#include "daxExecutive2.h"
#include "daxRegularArray.h"
#include "daxOptions.h"

#include <vector>
#include <string.h>
#include <assert.h>
#ifdef FNC_ENABLE_OPENCL
# include <CL/cl.hpp>
# include "opecl_util.h"
#endif

#include <boost/progress.hpp>

#define RETURN_ON_ERROR(err, msg) \
  {\
  if (err != CL_SUCCESS)\
    {\
    cerr << __FILE__<<":"<<__LINE__ << endl<< "ERROR("<<err<<"):  Failed to " << msg << endl;\
    cerr << "Error Code: " << oclErrorString(err) << endl;\
    return;\
    }\
  }

#define DIMENSION 256

#define uchar unsigned char
struct daxArrayCore
{
  uchar Type; // 0 -- irregular
              // 1 -- image-data points array
              // 2 -- image-data connections array
  uchar Rank;
  uchar Shape[2];
} __attribute__((__packed__));

struct daxImageDataData
{
  float Spacing[3];
  float Origin[3];
  unsigned int Extents[6];
} __attribute__((__packed__));

void daxExecute(int num_cores, daxArrayCore* cores,
  int num_in_arrays, float** in_arrays, size_t* in_arrays_size_in_bytes,
  int num_out_arrays, float** out_arrays, size_t* out_arrays_size_in_bytes,
  int num_kernels, const std::string* kernels)
{
#ifndef FNC_ENABLE_OPENCL
  cerr <<
    "You compiled without OpenCL support. So can't really execute "
    "anything. Here's the generated kernel. Have fun!" << endl;
#else
  // Now we should invoke the kernel using opencl setting up data arrays etc
  // etc.
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0)
    {
    cout << "No OpenCL capable platforms located." << endl;
    return;
    }

  cl_int err_code;
  try
    {
#ifdef __APPLE__
    cout << "Using CPU device" << endl;
    cl::Context context(CL_DEVICE_TYPE_CPU, NULL, NULL, NULL, &err_code);
#else
    cout << "Using GPU device" << endl;
    cl_context_properties properties[] =
      { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
    cl::Context context(CL_DEVICE_TYPE_GPU, properties);
#endif
    //RETURN_ON_ERROR(err_code, "create GPU Context");

    // Query devices.
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (devices.size()==0)
      {
      cout << "No OpenGL device located." << endl;
      return;
      }

    // Allocate buffer for cores.
    cl::Buffer arrayCores(context,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(daxArrayCore)*num_cores,
      cores, &err_code);
    RETURN_ON_ERROR(err_code, "upload array cores ptr");

    // Allocate input buffers.
    cl::Buffer **inputs  = new cl::Buffer*[num_in_arrays];
    for (int cc=0; cc < num_in_arrays; cc++)
      {
      inputs[cc] = new cl::Buffer(context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        in_arrays_size_in_bytes[cc],
        in_arrays[cc], &err_code);
      RETURN_ON_ERROR(err_code, "upload input data");
      }

    //Allocate output buffers.
    cl::Buffer **outputs = new cl::Buffer*[num_out_arrays];
    for (int cc=0; cc < num_out_arrays; cc++)
      {
      outputs[cc] = new cl::Buffer(context,
        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        out_arrays_size_in_bytes[cc],
        out_arrays[cc], &err_code);
      RETURN_ON_ERROR(err_code, "create output buffer");
      }

    // Now compile the code.
    cl::Program::Sources sources;
    sources.push_back(
      std::make_pair(daxHeaderString_dAPI, strlen(daxHeaderString_dAPI)));
    for (int cc=0; cc < num_kernels; cc++)
      {
      sources.push_back(std::make_pair(kernels[cc].c_str(), kernels[cc].size()));
      }

    // Build the code.
    cl::Program program (context, sources);
    err_code = program.build(devices);
    if (err_code != CL_SUCCESS)
      {
      std::string info;
      program.getBuildInfo(devices[0],
        CL_PROGRAM_BUILD_LOG, &info);
      cout << info.c_str() << endl;
      }
    RETURN_ON_ERROR(err_code, "compile the kernel.");

    cl::Kernel kernel(program, "entry_point", &err_code);
    RETURN_ON_ERROR(err_code, "locate entry-point 'main'.");

    // * determine the shape of the kernel invocation.

    // Kernel-shape will be decided by the output data type and the "head"
    // functor module.
    // For now, we simply invoke the kernel per item.

    // * pass arguments to the kernel

    err_code = kernel.setArg(0, arrayCores);
    RETURN_ON_ERROR(err_code, "pass array cores.");
    for (int cc=0; cc < num_in_arrays; cc++)
      {
      err_code = kernel.setArg(1+cc, *inputs[cc]);
      RETURN_ON_ERROR(err_code, "pass input buffer.");
      }
    for (int cc=0; cc < num_out_arrays; cc++)
      {
      err_code = kernel.setArg(1+num_in_arrays+cc, *outputs[cc]);
      RETURN_ON_ERROR(err_code, "pass output buffer.");
      }

    // FIXME num-of-cells.
    // FIXME: right now same as num-points since CellAverage is not really
    // computing cell-values currently.
    int num_items = ((DIMENSION-1)*(DIMENSION-1)*(DIMENSION-1));
    cout << num_items << endl;
    cl::Event event;
    cl::CommandQueue queue(context, devices[0]);
    err_code = queue.enqueueNDRangeKernel(kernel,
      cl::NullRange,
      cl::NDRange(num_items),
      cl::NullRange, NULL, &event);
    RETURN_ON_ERROR(err_code, "enqueue.");

    // for for the kernel execution to complete.
    boost::timer timer;
    err_code = event.wait();
    RETURN_ON_ERROR(err_code, "execute");
    cout << "Execution Time: " << timer.elapsed() << endl;

    // now request read back.
    for (int cc=0; cc < num_out_arrays; cc++)
      {
      err_code = queue.enqueueReadBuffer(*outputs[cc],
        CL_TRUE, 0,
        out_arrays_size_in_bytes[cc],
        out_arrays[cc]);
      RETURN_ON_ERROR(err_code, "read output back");
      }
    for (int cc=0; cc  < num_in_arrays; cc++)
      {
      delete inputs[cc];
      }
    delete [] inputs;
    for (int cc=0; cc  < num_out_arrays; cc++)
      {
      delete outputs[cc];
      }
    delete [] outputs;
    }
#ifdef __CL_ENABLE_EXCEPTIONS
  catch (cl::Error error)
    {
    cout << error.what() << "(" << error.err() << ")" << endl;
    }
#else
  catch (...)
    {
    cerr << "EXCEPTION"<< endl;
    return;
    }
#endif

#endif
}

#define USE_GRADIENT

int main(int, char**)
{
  daxExecutive2Ptr executive(new daxExecutive2());
  daxModulePtr elevation(new daxElevationModule());
#ifdef USE_GRADIENT
  daxModulePtr cellAverage(new daxCellGradientModule());
#else
  daxModulePtr cellAverage(new daxCellAverageModule());
#endif

  std::vector<std::string> kernels;
  kernels.push_back(elevation->GetCleandupFunctorCode());
  kernels.push_back(cellAverage->GetCleandupFunctorCode());

  executive->Connect(elevation, "output", cellAverage, "input_point");
  kernels.push_back(executive->GetKernel());

  // This pipeline has 5 arrays. 
  daxArrayCore cores[5];
  float* global_arrays[4];
  size_t global_array_size_in_bytes[4];

  // 0 == output of elevation.
  cores[0].Type = 0; 
  cores[0].Rank = 0;
  cores[0].Shape[0] = cores[0].Shape[1] = 0;

  // 1 == input to elevation (point coordinates)
  cores[1].Type = 1; // input image points coordinates
  cores[1].Rank = 1; // vectors
  cores[1].Shape[0] = 3; cores[1].Shape[1] = 0;
  daxImageDataData points;
  points.Spacing[0] = points.Spacing[1] = points.Spacing[2] = 1.0f;
  points.Origin[0] = points.Origin[1] = points.Origin[2] = 0.0f;
  points.Extents[0] = points.Extents[2] = points.Extents[4] = 0;
  points.Extents[1] = points.Extents[3] = points.Extents[5] = DIMENSION -1;
  global_arrays[0] = reinterpret_cast<float*>(&points);
  cout << "Size of points: " << sizeof(points) << endl;
  global_array_size_in_bytes[0] = sizeof(points);

  // 2 == output from cellAverage.
  cores[2].Type = 0; // irregular
  cores[2].Rank = 0;
  cores[2].Shape[0] = cores[2].Shape[1] = 0;
#ifdef USE_GRADIENT
  global_arrays[3] = new float[(DIMENSION-1)*(DIMENSION-1)*(DIMENSION-1) * 3];
  for (int cc=0; cc < (DIMENSION-1)*(DIMENSION-1)*(DIMENSION-1)*3; cc++)
    {
    global_arrays[3][cc] = -1;
    }
  global_array_size_in_bytes[3] =
    (DIMENSION-1)*(DIMENSION-1)*(DIMENSION-1)*sizeof(float)*3;
#else
  global_arrays[3] = new float[(DIMENSION-1)*(DIMENSION-1)*(DIMENSION-1)];
  for (int cc=0; cc < (DIMENSION-1)*(DIMENSION-1)*(DIMENSION-1); cc++)
    {
    global_arrays[3][cc] = -1;
    }
  global_array_size_in_bytes[3] = (DIMENSION-1)*(DIMENSION-1)*(DIMENSION-1)*sizeof(float);
#endif

  // 3 == same as 1 (point coordinates for CellAverage).
  cores[3] = cores[1];
  global_arrays[1] = global_arrays[0];
  global_array_size_in_bytes[1] = global_array_size_in_bytes[0];

  // 4 == cell-connections
  cores[4].Type = 2; // image cell
  cores[4].Rank = 1;
  cores[4].Shape[0] = 4; cores[4].Shape[1] = 0;
  global_arrays[2] = reinterpret_cast<float*>(&points);
  global_array_size_in_bytes[2] = sizeof(points);

  boost::timer timer;
  daxExecute(5, cores, 3, global_arrays, global_array_size_in_bytes,
    1, &global_arrays[3], &global_array_size_in_bytes[3],
    kernels.size(), &kernels[0]);
  cout << "Total Time ("<< DIMENSION << "^3) : "<< timer.elapsed() << endl;
  cout << "Output (1): " << global_arrays[3][0] << endl;
  delete []global_arrays[3];
  return 0;
}
