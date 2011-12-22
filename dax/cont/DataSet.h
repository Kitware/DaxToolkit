#ifndef __dax_cont_DataSet_h
#define __dax_cont_DataSet_h

#include <dax/cont/internal/Object.h>
#include <dax/cont/Array.h>
#include <dax/cont/FieldData.h>

namespace dax { namespace cont {
class DataSet : public dax::cont::internal::Object
{
public:
  typedef dax::cont::Array<dax::Vector3> Coordinates;
  typedef dax::cont::ArrayPtr<dax::Vector3> CoordinatesPtr;

  DataSet( const std::size_t& numPoints, const std::size_t& numCells);
  DataSet();

  virtual ~DataSet(){}

  virtual std::size_t numPoints() const { return NumPoints; }
  virtual std::size_t numCells() const { return NumCells; }

  virtual const CoordinatesPtr& points() const=0;

  FieldData& getFieldsPoint() { return FieldPoint; }
  const FieldData& getFieldsPoint() const { return FieldPoint; }

  FieldData& getFieldsCell() { return FieldCell; }
  const FieldData& getFieldsCell() const { return FieldCell; }

protected:

  std::size_t NumPoints;
  std::size_t NumCells;

  FieldData FieldPoint;
  FieldData FieldCell;
};

//------------------------------------------------------------------------------
DataSet::DataSet( const std::size_t& numPoints, const std::size_t& numCells):
  NumPoints(numPoints), NumCells(numCells)
{
}

//------------------------------------------------------------------------------
DataSet::DataSet():
  NumPoints(0), NumCells(0)
{
}


} }



#endif // __dax_cont_DataSet_h
