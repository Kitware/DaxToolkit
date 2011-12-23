/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_FieldData_h
#define __dax_cont_FieldData_h

#include <boost/static_assert.hpp>

#include <dax/cont/internal/Object.h>
#include <dax/cont/internal/FieldContainer.h>
#include <dax/cont/internal/ArrayContainer.h>
#include <dax/cont/Array.h>

namespace dax { namespace cont {
class FieldData : public dax::cont::internal::Object
{
private:
  typedef dax::cont::ArrayPtr<dax::Id> IdArray;
  typedef dax::cont::ArrayPtr<dax::Scalar> ScalarArray;
  typedef dax::cont::ArrayPtr<dax::Vector3> Vec3Array;
  typedef dax::cont::ArrayPtr<dax::Vector4> Vec4Array;

  typedef dax::cont::internal::ArrayContainer<dax::Id> IdContainer;
  typedef dax::cont::internal::ArrayContainer<dax::Scalar> ScalarContainer;
  typedef dax::cont::internal::ArrayContainer<dax::Vector3> Vec3Container;
  typedef dax::cont::internal::ArrayContainer<dax::Vector4> Vec4Container;

  dax::cont::internal::FieldContainer<dax::Id> IdField;
  dax::cont::internal::FieldContainer<dax::Scalar> ScalarField;
  dax::cont::internal::FieldContainer<dax::Vector3> Vec3Field;
  dax::cont::internal::FieldContainer<dax::Vector4> Vec4Field;

public:
  //Add array presumes you are adding the control
  //side of the array!
  bool addArray(const std::string& name, IdArray array)
    {return IdField.add(name,array);}
  bool addArray(const std::string& name, ScalarArray array)
    {return ScalarField.add(name,array);}
  bool addArray(const std::string& name, Vec3Array array)
    {return Vec3Field.add(name,array);}
  bool addArray(const std::string& name, Vec4Array array)
    {return Vec4Field.add(name,array);}

  bool addArray(const std::string& name,
                         const IdContainer& container)
    {return IdField.add(name,container);}
  bool addArray(const std::string& name,
                const ScalarContainer& container)
    {return ScalarField.add(name,container);}
  bool addArray(const std::string& name,
                const Vec3Container& container)
    {return Vec3Field.add(name,container);}
  bool addArray(const std::string& name,
                const Vec4Container& container)
    {return Vec4Field.add(name,container);}

  IdContainer& id(const std::string &name)
    { return IdField.get(name); }
  const IdContainer& id(const std::string &name) const
    { return IdField.get(name); }

  ScalarContainer& scalar(const std::string &name)
  { return ScalarField.get(name); }
  const ScalarContainer& scalar(const std::string &name) const
  { return ScalarField.get(name); }

  Vec3Container& vector3(const std::string &name)
  { return Vec3Field.get(name); }
  const Vec3Container& vector3(const std::string &name) const
  { return Vec3Field.get(name); }

  Vec4Container& vector4(const std::string &name)
  { return Vec4Field.get(name); }
  const Vec4Container& vector4(const std::string &name) const
  { return Vec4Field.get(name); }

  bool removeId(const std::string &name)
    { return IdField.remove(name); }
  bool removeScalar(const std::string &name)
  { return ScalarField.remove(name); }
  bool removeVector3(const std::string &name)
  { return Vec3Field.remove(name); }
  bool removeVector4(const std::string &name)
  { return Vec4Field.remove(name); }

  //Used to get the right field container based
  //on type ( id, scalar, vector3, or vector4 ).
  template <typename T>
  dax::cont::internal::ArrayContainer<T>& get(T t, const std::string &name);

  //Used to get the right field container based
  //on type ( id, scalar, vector3, or vector4 ).
  template <typename T>
  const dax::cont::internal::ArrayContainer<T>& get(T t,
                                                const std::string &name) const;
};


//------------------------------------------------------------------------------
template <typename T>
inline dax::cont::internal::ArrayContainer<T>& FieldData::get(
    T,const std::string& name)
{
  //we specialize id,scalar,vector3,and vector4 to call the right
  //internal containers get method. If a user attempts with any other
  //we fail at compile time

  //nvcc or gcc 4.4 tries to evaluate STATIC_ASSERT. So a workaround
  //is for the assert to be dependent on a template parameter
  BOOST_STATIC_ASSERT_MSG(sizeof(T)==0,NotAValidFieldType);

  //this return can never be hit
  return dax::cont::internal::ArrayContainer<T>();
}

//------------------------------------------------------------------------------
template <typename T>
inline const dax::cont::internal::ArrayContainer<T>& FieldData::get(
    T,const std::string& name) const
{
  //nvcc or gcc 4.4 tries to evaluate STATIC_ASSERT. So a workaround
  //is for the assert to be dependent on a template parameter
  BOOST_STATIC_ASSERT_MSG(sizeof(T)==0,NotAValidFieldType);

  //this return can never be hit
  return dax::cont::internal::ArrayContainer<T>();
}

//------------------------------------------------------------------------------
template <>
inline dax::cont::internal::ArrayContainer<dax::Id>& FieldData::get(
    dax::Id,const std::string& name)
{
  return this->id(name);
}

//------------------------------------------------------------------------------
template <>
inline dax::cont::internal::ArrayContainer<dax::Scalar>& FieldData::get(
    dax::Scalar,const std::string& name)
{
  return this->scalar(name);
}

//------------------------------------------------------------------------------
template <>
inline dax::cont::internal::ArrayContainer<dax::Vector3>& FieldData::get(
    dax::Vector3,const std::string& name)
{
  return this->vector3(name);
}

//------------------------------------------------------------------------------
template <>
inline dax::cont::internal::ArrayContainer<dax::Vector4>& FieldData::get(
    dax::Vector4, const std::string& name)
{
  return this->vector4(name);
}
//------------------------------------------------------------------------------
template <>
inline const dax::cont::internal::ArrayContainer<dax::Id>& FieldData::get(
    dax::Id,const std::string& name) const
{
  return this->id(name);
}

//------------------------------------------------------------------------------
template <>
inline const dax::cont::internal::ArrayContainer<dax::Scalar>& FieldData::get(
    dax::Scalar,const std::string& name) const
{
  return this->scalar(name);
}

//------------------------------------------------------------------------------
template <>
inline const dax::cont::internal::ArrayContainer<dax::Vector3>& FieldData::get(
    dax::Vector3,const std::string& name) const
{
  return this->vector3(name);
}

//------------------------------------------------------------------------------
template <>
inline const dax::cont::internal::ArrayContainer<dax::Vector4>& FieldData::get(
    dax::Vector4, const std::string& name) const
{
  return this->vector4(name);
}
} }

#endif // __dax_cont_FieldData_h

