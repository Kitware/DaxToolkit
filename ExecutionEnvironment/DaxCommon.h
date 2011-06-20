/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/// This file defines some common stuff.

#ifndef __DaxCommon_h
#define __DaxCommon_h

//*****************************************************************************
// Typedefs for basic types.
//*****************************************************************************
typedef float DaxScalar;
typedef float3 DaxVector3;
typedef float4 DaxVector4;
typedef int DaxId;

#define make_DaxVector3 make_float3
#define make_DaxVector4 make_float4

#define SUPERCLASS(__name__) \
  typedef __name__ Superclass

#endif
