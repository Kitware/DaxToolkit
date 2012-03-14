/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_Types_h
#define __dax_Types_h

#include <dax/internal/ExportMacros.h>

/*!
 * \namespace dax
 * \brief Dax Toolkit.
 *
 * dax is the namespace for the Dax Toolkit. It contains other sub namespaces,
 * as well as basic data types and functions callable from all components in
 * Dax toolkit.
 *
 * \namespace dax::cont
 * \brief Dax Control Environment.
 *
 * dax::cont defines the publicly accessible API for the Dax Control
 * Environment. Users of the Dax Toolkit can use this namespace to access the
 * Control Environment.
 *
 * \namespace dax::exec
 * \brief Dax Execution Environment.
 *
 * dax::exec defines the publicly accessible API for the Dax Execution
 * Environment. Worklets typically use classes/apis defined within this
 * namespace alone.
 *
 * \namespace dax::cuda
 * \brief CUDA implementation.
 *
 * dax::cuda includes the code to implement the Dax for CUDA-based
 * platforms.
 *
 * \namespace dax::cuda::cont
 * \brief CUDA implementation for Control Environment.
 *
 * dax::cuda::cont includes the code to implement the Dax Control Environment
 * for CUDA-based platforms.
 *
 * \namespace dax::cuda::exec
 * \brief CUDA implementation for Execution Environment.
 *
 * dax::cuda::exec includes the code to implement the Dax Execution Environment
 * for CUDA-based platforms.
 *
 * \namespace dax::testing
 * \brief Internal testing classes
 *
 */

namespace dax
{
//*****************************************************************************
// Typedefs for basic types.
//*****************************************************************************

/// Alignment requirements are prescribed by CUDA on device (Table B-1 in NVIDIA
/// CUDA C Programming Guide 4.0)

/// Scalar corresponds to a single-valued floating point number.
typedef float Scalar __attribute__ ((aligned (4)));

/// Represents an ID.
typedef int Id __attribute__ ((aligned(4)));

/// Tuple corresponds to a Size-tuple of type T
template<typename T, int Size>
class Tuple {
public:
  typedef T ValueType;
  static const int NUM_COMPONENTS=Size;

  DAX_EXEC_CONT_EXPORT Tuple(){}
  DAX_EXEC_CONT_EXPORT explicit Tuple(const ValueType& value)
    {
    for(int i=0; i < NUM_COMPONENTS;++i)
      {
      this->Values[i]=value;
      }
    }
  DAX_EXEC_CONT_EXPORT explicit Tuple(const ValueType* values)
    {
    for(int i=0; i < NUM_COMPONENTS;++i)
      {
      this->Values[i]=values[i];
      }
    }

  DAX_EXEC_CONT_EXPORT const ValueType &operator[](int idx) const {
    return this->Values[idx];
  }
  DAX_EXEC_CONT_EXPORT ValueType &operator[](int idx) {
    return this->Values[idx];
  }

protected:
  ValueType Values[NUM_COMPONENTS];
}__attribute__ ((aligned(4)));


/// Vector3 corresponds to a 3-tuple
class Vector3 : public dax::Tuple<dax::Scalar,3>{
public:

  DAX_EXEC_CONT_EXPORT Vector3() {}
  DAX_EXEC_CONT_EXPORT explicit Vector3(const dax::Scalar& value):
    dax::Tuple<dax::Scalar, 3>(value){}
  DAX_EXEC_CONT_EXPORT explicit Vector3(const dax::Scalar* values):
    dax::Tuple<dax::Scalar, 3>(values) { }

  DAX_EXEC_CONT_EXPORT Vector3(ValueType x, ValueType y, ValueType z) {
    this->Values[0] = x;
    this->Values[1] = y;
    this->Values[2] = z;
  }
} __attribute__ ((aligned(4)));

/// Vector4 corresponds to a 4-tuple
class Vector4 : public dax::Tuple<Scalar,4>{
public:

  DAX_EXEC_CONT_EXPORT Vector4() {}
  DAX_EXEC_CONT_EXPORT explicit Vector4(const dax::Scalar& value):
    dax::Tuple<dax::Scalar, 4>(value){}
  DAX_EXEC_CONT_EXPORT explicit Vector4(const dax::Scalar* values):
    dax::Tuple<dax::Scalar, 4>(values) { }

  DAX_EXEC_CONT_EXPORT
  Vector4(ValueType x, ValueType y, ValueType z, ValueType w) {
    this->Values[0] = x;
    this->Values[1] = y;
    this->Values[2] = z;
    this->Values[3] = w;
  }
} __attribute__ ((aligned(4)));


/// Id3 corresponds to a 3-dimensional index for 3d arrays.  Note that
/// the precision of each index may be less than dax::Id.
class Id3 : public dax::Tuple<dax::Id,3>{
public:

  DAX_EXEC_CONT_EXPORT Id3() {}
  DAX_EXEC_CONT_EXPORT explicit Id3(const dax::Id& value):
    dax::Tuple<dax::Id, 3>(value){}
  DAX_EXEC_CONT_EXPORT explicit Id3(const dax::Id* values):
    dax::Tuple<dax::Id, 3>(values) { }

  DAX_EXEC_CONT_EXPORT Id3(ValueType x, ValueType y, ValueType z) {
    this->Values[0] = x;
    this->Values[1] = y;
    this->Values[2] = z;
  }
} __attribute__ ((aligned(4)));

/// Initializes and returns a Vector3.
DAX_EXEC_CONT_EXPORT dax::Vector3 make_Vector3(dax::Scalar x,
                                               dax::Scalar y,
                                               dax::Scalar z)
{
  return dax::Vector3(x, y, z);
}

/// Initializes and returns a Vector4.
DAX_EXEC_CONT_EXPORT dax::Vector4 make_Vector4(dax::Scalar x,
                                               dax::Scalar y,
                                               dax::Scalar z,
                                               dax::Scalar w)
{
  return dax::Vector4(x, y, z, w);
}

/// Initializes and returns an Id3
DAX_EXEC_CONT_EXPORT dax::Id3 make_Id3(dax::Id x, dax::Id y, dax::Id z)
{
  return dax::Id3(x, y, z);
}

DAX_EXEC_CONT_EXPORT dax::Id dot(dax::Id a, dax::Id b)
{
  return a * b;
}

DAX_EXEC_CONT_EXPORT dax::Id3::ValueType dot(const dax::Id3 &a,
                                             const dax::Id3 &b)
{
  return (a[0]*b[0]) + (a[1]*b[1]) + (a[2]*b[2]);
}

DAX_EXEC_CONT_EXPORT dax::Scalar dot(dax::Scalar a, dax::Scalar b)
{
  return a * b;
}

DAX_EXEC_CONT_EXPORT dax::Vector3::ValueType dot(const dax::Vector3 &a,
                                                 const dax::Vector3 &b)
{
  return (a[0]*b[0]) + (a[1]*b[1]) + (a[2]*b[2]);
}

DAX_EXEC_CONT_EXPORT dax::Vector4::ValueType dot(const dax::Vector4 &a,
                                                 const dax::Vector4 &b)
{
  return (a[0]*b[0]) + (a[1]*b[1]) + (a[2]*b[2]) + (a[3]*b[3]);
}

} // End of namespace dax

DAX_EXEC_CONT_EXPORT dax::Id3 operator+(const dax::Id3 &a,
                                        const dax::Id3 &b)
{
  return dax::make_Id3(a[0]+b[0], a[1]+b[1], a[2]+b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Id3 operator*(const dax::Id3 &a,
                                        const dax::Id3 &b)
{
  return dax::make_Id3(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Id3 operator-(const dax::Id3 &a,
                                        const dax::Id3 &b)
{
  return dax::make_Id3(a[0]-b[0], a[1]-b[1], a[2]-b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Id3 operator/(const dax::Id3 &a,
                                        const dax::Id3 &b)
{
  return dax::make_Id3(a[0]/b[0], a[1]/b[1], a[2]/b[2]);
}
DAX_EXEC_CONT_EXPORT bool operator==(const dax::Id3 &a,
                                     const dax::Id3 &b)
{
  return (a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]);
}
DAX_EXEC_CONT_EXPORT bool operator!=(const dax::Id3 &a,
                                     const dax::Id3 &b)
{
  return !(a == b);
}

DAX_EXEC_CONT_EXPORT dax::Id3 operator*(dax::Id3::ValueType a,
                                        const dax::Id3 &b)
{
  return dax::make_Id3(a*b[0], a*b[1], a*b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Id3 operator*(const dax::Id3 &a,
                                        dax::Id3::ValueType &b)
{
  return dax::make_Id3(a[0]*b, a[1]*b, a[2]*b);
}

DAX_EXEC_CONT_EXPORT dax::Vector3 operator+(const dax::Vector3 &a,
                                            const dax::Vector3 &b)
{
  return dax::make_Vector3(a[0]+b[0], a[1]+b[1], a[2]+b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 operator*(const dax::Vector3 &a,
                                            const dax::Vector3 &b)
{
  return dax::make_Vector3(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 operator-(const dax::Vector3 &a,
                                            const dax::Vector3 &b)
{
  return dax::make_Vector3(a[0]-b[0], a[1]-b[1], a[2]-b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 operator/(const dax::Vector3 &a,
                                            const dax::Vector3 &b)
{
  return dax::make_Vector3(a[0]/b[0], a[1]/b[1], a[2]/b[2]);
}
DAX_EXEC_CONT_EXPORT bool operator==(const dax::Vector3 &a,
                                     const dax::Vector3 &b)
{
  return (a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]);
}
DAX_EXEC_CONT_EXPORT bool operator!=(const dax::Vector3 &a,
                                     const dax::Vector3 &b)
{
  return !(a == b);
}

DAX_EXEC_CONT_EXPORT dax::Vector3 operator*(dax::Vector3::ValueType a,
                                            const dax::Vector3 &b)
{
  return dax::make_Vector3(a*b[0], a*b[1], a*b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 operator*(const dax::Vector3 &a,
                                            dax::Vector3::ValueType &b)
{
  return dax::make_Vector3(a[0]*b, a[1]*b, a[2]*b);
}

DAX_EXEC_CONT_EXPORT dax::Vector4 operator+(const dax::Vector4 &a,
                                            const dax::Vector4 &b)
{
  return dax::make_Vector4(a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 operator*(const dax::Vector4 &a,
                                            const dax::Vector4 &b)
{
  return dax::make_Vector4(a[0]*b[0], a[1]*b[1], a[2]*b[2], a[3]*b[3]);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 operator-(const dax::Vector4 &a,
                                            const dax::Vector4 &b)
{
  return dax::make_Vector4(a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 operator/(const dax::Vector4 &a,
                                            const dax::Vector4 &b)
{
  return dax::make_Vector4(a[0]/b[0], a[1]/b[1], a[2]/b[2], a[3]/b[3]);
}
DAX_EXEC_CONT_EXPORT bool operator==(const dax::Vector4 &a,
                                     const dax::Vector4 &b)
{
  return (a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]) && (a[3] == b[3]);
}
DAX_EXEC_CONT_EXPORT bool operator!=(const dax::Vector4 &a,
                                     const dax::Vector4 &b)
{
  return !(a == b);
}

DAX_EXEC_CONT_EXPORT dax::Vector4 operator*(dax::Vector4::ValueType a,
                                            const dax::Vector4 &b)
{
  return dax::make_Vector4(a*b[0], a*b[1], a*b[2], a*b[3]);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 operator*(const dax::Vector4 &a,
                                            dax::Scalar &b)
{
  return dax::make_Vector4(a[0]*b, a[1]*b, a[2]*b, a[3]*b);
}

#endif
