//-----------------------------------------------------------------------------
// Defines Data Types for DAX.
// Conventions:
// * All function names should be prefixed with "dax" and distinguish words in
//   camel case. The current API already does this.
// * Words in the name should always go from general groups to specific. For
//   example, use "daxMatrixAdd" and "daxMatrixMultiply" instead of
//   "daxAddMatrices" and "daxMultiplyMatrices." Although this sometimes leads
//   to more awkward phrasing, it makes it much easier to find functions in
//   alphabetized documentation and easier to predict how functions are named.
// * Functions that operate on an object of a particular type should identify
//   the name of the object right after "dax." Thus, any function that operates
//   on a work object will start with daxWork, and function that operates on a
//   cell object will start with daxCell, etc. This rule is basically a
//   refinement of the previous rule and makes sure to avoid name clashes.
// * Functions that get or set state in an object should be named
//   "dax<object-type>Get<data-name>" or likewise for Set. For example,
//   "daxCellGetNumberOfPoints" and "daxArraySetValue3." The rational follows
//   that of the previous rule.
// * Never use "Get" or "Set" in a function that does not access the state of an
//   object. Don’t use "Get" when computing values. For example, use
//   "daxCellDerivative" instead of "daxGetCellDerivative.
// * Internal functions are prefixed by a single "_" e.g. _daxWorkNew. Worklets
//   should  never use internal functions.
//-----------------------------------------------------------------------------

#include "DaxDataTypes.cu"

//*****************************************************************************
// Functions to operate with Work.
//*****************************************************************************

//-----------------------------------------------------------------------------
__device__ DaxCell daxWorkGetCell(const DaxWorkMapCell work)
{
  DaxCell cell;
  cell.Work = work;
  return cell;
}

//-----------------------------------------------------------------------------
__device__ DaxDualCell daxWorkGetDualCell(const DaxWorkMapPoint work)
{
  DaxCell cell;
  cell.Work = work;
  return cell;
}

//*****************************************************************************
// Functions that get/set values from DaxFields
//*****************************************************************************

//-----------------------------------------------------------------------------
// Returns the number of components for a field.
__device__ DaxId daxWorkGetScalarNumberOfComponents(
  const _DaxWork work, const _DaxWork field)
{
  return 0;
}

//-----------------------------------------------------------------------------
// Returns the scalar for a component of a field.
__device__ DaxScalar daxWorkGetScalarValue(const _DaxWork work,
  const _DaxField field, const DaxId component_no)
{
  DaxScalar foo = 0.0;
  return foo;
}

//-----------------------------------------------------------------------------
// Returns the first 3 components from a vector field.
__device__ DaxVector3 daxWorkGetVectorValue3(
  const _DaxWork work, const _DaxField field)
{
  DaxVector3 vec3 = make_DaxVector3(0, 0, 0);
  return vec3;
}

//-----------------------------------------------------------------------------
// Returns the first 4 components from a vector field.
__device__ DaxVector4 daxWorkGetVectorValue4(
  const _DaxWork work, const _DaxField field)
{
  DaxVector4 vec4 = make_DaxVector4(0, 0, 0, 0);
  return vec4;
}

//-----------------------------------------------------------------------------
// Returns the magnitude for the vector field.
__device__ DaxScalar daxWorkGetVectorMagnitude(
  const _DaxWork work, const _DaxField field)
{
  return 0;
}

//*****************************************************************************
// Utility functions for operating with topological units.
//*****************************************************************************

//-----------------------------------------------------------------------------
__device__ DaxVector3 daxCellDerivative(
  const DaxCell cell,
  const DaxVector3 parametric_point,
  const DaxFieldCoordinates positions,
  const DaxFieldPoint scalar,
  DaxId scalar_component)
{
  DaxVector3 vec3 = make_DaxVector3(0, 0, 0);
  return vec3;
}

//-----------------------------------------------------------------------------
__device__ DaxScalar daxCellInterpolate(
  const DaxCell cell,
  const DaxVector3 parametric_point,
  const DaxFieldPoint scalar,
  DaxId scalar_component)
{
  return 0;
}

//-----------------------------------------------------------------------------
__device__ DaxScalar daxDualCellInterpolate(
  const DaxDualCell dual_cell,
  const DaxVector3 parametric_dual_point,
  const DaxFieldCell scalar,
  DaxId scalar_component)
{
  return 0;
}

