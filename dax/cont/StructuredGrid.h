/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_StructuredGrid_h
#define __dax_cont_StructuredGrid_h

#include <dax/cont/DataSet.h>

namespace dax { namespace cont {

class StructuredGrid : public dax::cont::DataSet
{
private:
  struct Extent3 {
    Id3 Min;
    Id3 Max;
  };

public:  
  StructuredGrid();
  StructuredGrid(const dax::Vector3 &origin,
                 const dax::Vector3 &spacing,
                 const dax::Id3 &minExtents,
                 const dax::Id3 &maxExtents);

  virtual ~StructuredGrid();

  virtual std::size_t numPoints() const;
  virtual std::size_t numCells() const;

  const Coordinates& points() const { return Points; }
  Coordinates& points() { return Points;}

  void computePointLocations();

  Vector3 Origin;
  Vector3 Spacing;
  Extent3 Extent;
private:
  Coordinates Points;

  dax::Id3 extentDimensions() const;
  dax::Id numberOfPoints() const;
  dax::Id numberOfCells() const;
};

//------------------------------------------------------------------------------
StructuredGrid::StructuredGrid():
  DataSet(), Points()
{

  this->Origin = dax::make_Vector3(0,0,0);
  this->Spacing = dax::make_Vector3(0,0,0);
  this->Extent.Min = dax::make_Id3(0,0,0);
  this->Extent.Max = dax::make_Id3(0,0,0);
}

//------------------------------------------------------------------------------
StructuredGrid::StructuredGrid(const dax::Vector3 &origin,
                               const dax::Vector3 &spacing,
                               const dax::Id3 &minExtents,
                               const dax::Id3 &maxExtents):
  DataSet(), Points()
{
  this->Origin = origin;
  this->Spacing = spacing;
  this->Extent.Min = minExtents;
  this->Extent.Max = maxExtents;
}

//------------------------------------------------------------------------------
StructuredGrid::~StructuredGrid()
{
}

//------------------------------------------------------------------------------
std::size_t StructuredGrid::numPoints() const
{
  return this->numberOfPoints();
}

//------------------------------------------------------------------------------
std::size_t StructuredGrid::numCells() const
{
  return this->numberOfCells();
}

//------------------------------------------------------------------------------
void StructuredGrid::computePointLocations()
{
  std::size_t size = this->numPoints();
  dax::cont::ArrayPtr<Coordinates::ValueType> realPoints(
        new dax::cont::Array<Coordinates::ValueType>());
  realPoints->resize(size);
  for (std::size_t i=0; i < size; ++i)
    {
    (*realPoints)[i] = dax::make_Vector3(i,i,i);
    }
  //set the control array for the points
  this->Points.setArrayControl(realPoints);
}

//------------------------------------------------------------------------------
dax::Id3 StructuredGrid::extentDimensions() const
{
  return Extent.Max - Extent.Min + dax::make_Id3(1, 1, 1);
}

//------------------------------------------------------------------------------
dax::Id StructuredGrid::numberOfPoints() const
{
  dax::Id3 dims = this->extentDimensions();
  return dims[0]*dims[1]*dims[2];
}

//------------------------------------------------------------------------------
dax::Id StructuredGrid::numberOfCells() const
{
  dax::Id3 dims = this->extentDimensions() - dax::make_Id3(1, 1, 1);
  return dims[0]*dims[1]*dims[2];
}

} }
#endif // __dax_cont_StructuredGrid_h
