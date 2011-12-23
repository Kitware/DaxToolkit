/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_DataSet_h
#define __dax_cont_DataSet_h

#include <dax/cont/internal/Object.h>
#include <dax/cont/internal/ArrayContainer.h>
#include <dax/cont/FieldData.h>

namespace dax { namespace cont {
class DataSet : public dax::cont::internal::Object
{
public:
  typedef dax::cont::internal::ArrayContainer<dax::Vector3> Coordinates;

  DataSet( const std::size_t& numPoints, const std::size_t& numCells);
  DataSet();

  virtual ~DataSet(){}

  virtual std::size_t numPoints() const { return NumPoints; }
  virtual std::size_t numCells() const { return NumCells; }

  virtual const Coordinates& points() const=0;
  virtual Coordinates& points()=0;

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
