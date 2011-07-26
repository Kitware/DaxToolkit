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
private:
  unsigned int MaxWarpSize;
  unsigned int MaxGridSize;
  unsigned int ProblemSize;
};
