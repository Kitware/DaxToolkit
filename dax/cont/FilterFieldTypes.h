/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_FilterFieldTypes_h
#define __dax_cont_FilterFieldTypes_h
#include <dax/internal/GridStructures.h>
#include <string>
namespace dax { namespace cont {


//todo find a better way than these
class FieldType
{
public:
  enum TYPE{ POINTS=0, CELLS=1 };

 virtual std::string name() const =0;
 virtual dax::cont::FieldType::TYPE type() const=0;
};

class PointField : public FieldType
{
public:
  virtual std::string name() const { return "Points"; }
  virtual dax::cont::FieldType::TYPE type() const { return FieldType::POINTS; }

  template <typename T>
  static dax::Id size(T *t) { return dax::internal::numberOfPoints(*t); }
};
class CellField : public FieldType
{
public:
  virtual std::string name() const { return "Cells"; }
  virtual dax::cont::FieldType::TYPE type() const { return FieldType::CELLS; }

  template <typename T>
  static dax::Id size(T *t) { return dax::internal::numberOfCells(*t); }
};


class DeviceType
{
public:
  virtual std::string name()=0;
};

class CudaDevice : public DeviceType
{
public:
  virtual std::string name() { return "Cuda"; }
};

class OpenMPDevice : public DeviceType
{
public:
  virtual std::string name() { return "OpenMP"; }
};


template<typename F, typename T>
dax::Id FieldSize(F *field,T *t )
{
  if(field->type()== FieldType::POINTS)
    {
    return dax::cont::PointField::size(t);
    }
  else if(field->type() == FieldType::CELLS)
    {
    return dax::cont::CellField::size(t);
    }
  return -1;
}


} }

#endif
