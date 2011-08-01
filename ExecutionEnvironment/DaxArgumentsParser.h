/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

class DaxArgumentsParser
{
public:
  DaxArgumentsParser();
  virtual ~DaxArgumentsParser();

  bool ParseArguments(int argc, char* argv[]);

  unsigned int GetMaxWarpSize() const
    { return this->MaxWarpSize; }
  unsigned int GetMaxGridSize() const
    { return this->MaxGridSize; }
  unsigned int GetProblemSize() const
    { return this->ProblemSize; }

  enum PipelineMode
    {
    CELL_GRADIENT = 1,
    CELL_GRADIENT_SINE_SQUARE_COS = 2,
    SINE_SQUARE_COS = 3,
    };
  PipelineMode GetPipeline() const
    { return this->Pipeline; }
private:
  unsigned int MaxWarpSize;
  unsigned int MaxGridSize;
  unsigned int ProblemSize;
  PipelineMode Pipeline;
};
