//-----------------------------------------------------------------------------
// Defines Data Types for DAX.
// Conventions:
// * Types should be named like functions with camel case identifiers. However,
//   they should start with "Dax" instead of "dax" to differentiate them.
// * Like functions, the words in type names should go from general groups to
//   specific. For example, "DaxCellPolygon" instead of "DaxPolygonCell".
// * All types for opaque objects (i.e. for work, cell, etc.) should encapsulate
//   any pointer within it. There should be no reason to ever have a pointer to
//   DaxCell. Thus, objects should not need "*" in their declaration or need "&"
//   when calling a function.
// * Internal types are prefixed by a single "_" e.g. _DaxWork. Worklets should
//   never use internal types.
//-----------------------------------------------------------------------------

//*****************************************************************************
// Typedefs for basic types.
//*****************************************************************************
typedef float DaxScalar;
typedef float3 DaxVector3;
typedef float4 DaxVector4;
typedef int DaxId;

//*****************************************************************************
// Compound Types
//*****************************************************************************

//-----------------------------------------------------------------------------
// Types used to identify "work" for a worklet.
class _DaxWork
{
public:
  __device__ __host__ _DaxWork()
    {
    this->ElementId = (int)(threadIdx.x);
    }
  DaxId ElementId;
};

typedef _DaxWork* _DaxWorkPtr;
typedef _DaxWorkPtr DaxWorkMapField;
typedef _DaxWorkPtr DaxWorkMapCell;
typedef _DaxWorkPtr DaxWorkMapPoint;

#define make_DaxVector3 make_float3
#define make_DaxVector4 make_float4

//-----------------------------------------------------------------------------
// Types used to describe data fields for a worklet.

typedef struct
{
  // Must have attributes to be able to tell which "global-memory" array it
  // corresponds to.
  
} _DaxField;

typedef _DaxField DaxField;
typedef _DaxField DaxFieldPoint;
typedef _DaxField DaxFieldCell;
typedef _DaxField DaxFieldCoordinates;

//-----------------------------------------------------------------------------

typedef struct
{
  _DaxWorkPtr Work;
} _DaxCell;

typedef _DaxCell DaxCell;

typedef _DaxCell DaxDualCell;

//*****************************************************************************
// MACROS/MODIFIERS
//*****************************************************************************
#define DAX_IN
#define DAX_OUT const
#define DAX_WORKLET __device__
