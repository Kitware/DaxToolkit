// Simple executable to send an an OpenCL file and it builds the kernel.

#include "daxSystemIncludes.h"
#include "daxOptions.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <fstream>
#include <iterator>
#include <vector>
#include <string>
using namespace std;

#include <vector>
#include <string.h>
#include <assert.h>
#ifdef FNC_ENABLE_OPENCL
# include <CL/cl.hpp>
# include "opecl_util.h"
#endif

#define RETURN_ON_ERROR(err, msg) \
  {\
  if (err != CL_SUCCESS)\
    {\
    cerr << __FILE__<<":"<<__LINE__ << endl<< "ERROR("<<err<<"):  Failed to " << msg << endl;\
    cerr << "Error Code: " << oclErrorString(err) << endl;\
    return -1;\
    }\
  }

int main(int argc, char** argv)
{
  po::options_description desc("Allowed options");
  desc.add_options()
    ("input-file", po::value< vector<string> >(), "Input files (required)")
    ("help", "Generate this help message");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help") != 0 ||
    vm.count("input-file") == 0)
    {
    cout << desc << endl;
    return 1;
    }

  vector<string> opencl_codes;
  vector<string> input_files = vm["input-file"].as< vector<string> >();
  for (size_t cc=0; cc < input_files.size(); cc++)
    {
    ifstream is(input_files[cc].c_str());
    if (is.is_open())
      {
      // get length of file:
      is.seekg (0, ios::end);
      size_t length = is.tellg();
      is.seekg (0, ios::beg);

      // allocate memory:
      char* buffer = new char [length + 1];

      // read data as a block:
      is.read (buffer,length);
      buffer[length] = 0;
      opencl_codes.push_back(string(buffer));
      delete[] buffer;
      }
    is.close();
    }

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
      return -1;
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
        return -1;
        }

      // Now compile the code.
      cl::Program::Sources sources;
      for (size_t cc=0; cc < opencl_codes.size(); cc++)
        {
        sources.push_back(std::make_pair(opencl_codes[cc].c_str(), opencl_codes[cc].size()));
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
      return -1;
      }
#endif

#endif
    }
  return 0;
}
