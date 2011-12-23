#ifndef __dax_cont_StructuredGrid_h
#define __dax_cont_StructuredGrid_h

#include <dax/cont/DataSet.h>

namespace dax { namespace cont {

class StructuredGrid :
    public dax::cont::DataSet
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

  Vector3 Origin;
  Vector3 Spacing;
  Extent3 Extent;


  void computePointLocations()
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

  const Coordinates& points() const
    {
    return Points;
    }


  Coordinates& points()
    {
    return Points;
    }

  virtual ~StructuredGrid();

  virtual std::size_t numPoints() const;
  virtual std::size_t numCells() const;

private:
  Coordinates Points;

  dax::Id3 extentDimensions() const
  {
    return Extent.Max - Extent.Min + dax::make_Id3(1, 1, 1);
  }

  dax::Id numberOfPoints() const
  {
    dax::Id3 dims = this->extentDimensions();
    return dims.x*dims.y*dims.z;
  }

  dax::Id numberOfCells() const
  {
    dax::Id3 dims = this->extentDimensions() - dax::make_Id3(1, 1, 1);
    return dims.x*dims.y*dims.z;
  }


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


} }
#endif // __dax_cont_StructuredGrid_h
