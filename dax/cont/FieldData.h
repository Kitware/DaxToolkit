/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_FieldData_h
#define __dax_cont_FieldData_h

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

  IdContainer& getId(const std::string &name)
    { return IdField.get(name); }
  const IdContainer& getId(const std::string &name) const
    { return IdField.get(name); }

  ScalarContainer& getScalar(const std::string &name)
  { return ScalarField.get(name); }
  const ScalarContainer& getScalar(const std::string &name) const
  { return ScalarField.get(name); }

  Vec3Container& getVector3(const std::string &name)
  { return Vec3Field.get(name); }
  const Vec3Container& getVector3(const std::string &name) const
  { return Vec3Field.get(name); }

  Vec4Container& getVector4(const std::string &name)
  { return Vec4Field.get(name); }
  const Vec4Container& getVector4(const std::string &name) const
  { return Vec4Field.get(name); }

  bool removeId(const std::string &name)
    { return IdField.remove(name); }
  bool removeScalar(const std::string &name)
  { return ScalarField.remove(name); }
  bool removeVector3(const std::string &name)
  { return Vec3Field.remove(name); }
  bool removeVector4(const std::string &name)
  { return Vec4Field.remove(name); }
};
} }

#endif // __dax_cont_FieldData_h

