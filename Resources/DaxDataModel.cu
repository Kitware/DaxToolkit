//-----------------------------------------------------------------------------
// Defines data model in the execution environment.
//-----------------------------------------------------------------------------
#include "DaxDataTypes.cu"

#define DaxSetMacro(_name, _type)\
  __device__ __host__ void Set##_name(_type __val)\
  { this->_name = __val; }

#define DaxGetMacro(_name, _type)\
  __device__ __host__ _type Get##_name() const { return this->_name; }

class _DaxArray
{
protected:
  int NumberOfComponents;
  int NumberOfTuples;
  unsigned char Type;
  void* Data;
public:
  _DaxArray() :
    NumberOfComponents(0), NumberOfTuples(0), Type(0), Data(NULL) { }

  /// Get/Set the number of tuples.
  DaxSetMacro(NumberOfTuples, int);
  DaxGetMacro(NumberOfTuples, int);

  /// Get/Set the number of components.
  DaxSetMacro(NumberOfComponents, int);
  DaxGetMacro(NumberOfComponents, int);

  __device__ virtual DaxScalar GetScalar(DaxId offset) const
    { return 0; }
};

class _DaxArrayIrregular : public _DaxArray
{
public:
  _DaxArrayIrregular()
    {
    this->Type = 1;
    }
};
