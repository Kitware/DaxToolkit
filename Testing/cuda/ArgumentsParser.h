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
    CELL_GRADIENT = 1,
    CELL_GRADIENT_SINE_SQUARE_COS = 2,
    SINE_SQUARE_COS = 3
    };
  PipelineMode pipeline() const
    { return this->Pipeline; }
private:
  unsigned int ProblemSize;
  PipelineMode Pipeline;
};

}}
