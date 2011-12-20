#ifndef __dax_cont_FieldData_h
#define __dax_cont_FieldData_h

#include <dax/cont/internal/Object.h>
#include <dax/cont/internal/FieldContainer.h>
#include <dax/cont/Array.h>

namespace dax { namespace cont {
class FieldData : public dax::cont::internal::Object
{
private:
  typedef dax::cont::Array<dax::Id> IdArray;
  typedef dax::cont::Array<dax::Scalar> ScalarArray;
  typedef dax::cont::Array<dax::Vector3> Vec3Array;
  typedef dax::cont::Array<dax::Vector4> Vec4Array;

  dax::cont::internal::FieldContainer<dax::Id> IdContainer;
  dax::cont::internal::FieldContainer<dax::Scalar> ScalarContainer;
  dax::cont::internal::FieldContainer<dax::Vector3> Vec3Container;
  dax::cont::internal::FieldContainer<dax::Vector4> Vec4Container;

public:
  bool addArray(const std::string& name, IdArray &array)
    {return IdContainer.add(name,array);}
  bool addArray(const std::string& name, ScalarArray &array)
    {return ScalarContainer.add(name,array);}
  bool addArray(const std::string& name, Vec3Array &array)
    {return Vec3Container.add(name,array);}
  bool addArray(const std::string& name, Vec4Array &array)
    {return Vec4Container.add(name,array);}

  IdArray &getId(const std::string &name)
    { return IdContainer.get(name); }
  const IdArray &getId(const std::string &name) const
    { return IdContainer.get(name); }
  void getId(const std::string &name, IdArray& array) const
    { return IdContainer.get(name,array); }

  ScalarArray &getScalar(const std::string &name)
  { return ScalarContainer.get(name); }
  const ScalarArray &getScalar(const std::string &name) const
  { return ScalarContainer.get(name); }
  void getScalar(const std::string &name, ScalarArray& array) const
  { return ScalarContainer.get(name,array); }

  Vec3Array &getVector3(const std::string &name)
  { return Vec3Container.get(name); }
  const Vec3Array &getVector3(const std::string &name) const
  { return Vec3Container.get(name); }
  void getVector3(const std::string &name, Vec3Array& array) const
  { return Vec3Container.get(name,array); }

  Vec4Array &getVector4(const std::string &name)
  { return Vec4Container.get(name); }
  const Vec4Array &getVector4(const std::string &name) const
  { return Vec4Container.get(name); }
  void getVector4(const std::string &name, Vec4Array& array) const
  { return Vec4Container.get(name,array); }

  bool removeId(const std::string &name)
    { return IdContainer.remove(name); }
  bool removeScalar(const std::string &name)
  { return ScalarContainer.remove(name); }
  bool removeVector3(const std::string &name)
  { return Vec3Container.remove(name); }
  bool removeVector4(const std::string &name)
  { return Vec4Container.remove(name); }
};
} }

#endif // __dax_cont_FieldData_h

