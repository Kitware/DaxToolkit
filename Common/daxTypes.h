/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxTypes_h
#define __daxTypes_h

/// Alignment requirements are prescribed by CUDA on device (Table B-1 in NVIDIA
/// CUDA C Programming Guide 4.0)
typedef float DaxScalar __attribute__ ((aligned (4)));
typedef struct { float x; float y; float z; } DaxVector3 __attribute__
((aligned(4)));
typedef struct { float x; float y; float z; float w; } DaxVector4 __attribute__
((aligned(16)));

typedef int DaxId __attribute__ ((aligned(4)));
#endif
