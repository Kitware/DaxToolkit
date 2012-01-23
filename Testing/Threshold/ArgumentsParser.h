/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

namespace dax { namespace testing {

class ArgumentsParser
{
public:
  ArgumentsParser();
  virtual ~ArgumentsParser();

  bool parseArguments(int argc, char* argv[]);

  unsigned int problemSize() const
    { return this->ProblemSize; }

  enum PipelineMode
    {
    CELL_THRESHOLD = 1
    };
  PipelineMode pipeline() const
    { return this->Pipeline; }

  enum DeviceAdapterMode
  {
    DEVICE_ALL,
    DEVICE_DEBUG,
    DEVICE_OPENMP,
    DEVICE_CUDA
  };
  DeviceAdapterMode device() const { return this->Device; }

private:
  unsigned int ProblemSize;
  PipelineMode Pipeline;
  DeviceAdapterMode Device;
};

}}
