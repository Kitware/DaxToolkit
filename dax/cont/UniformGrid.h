/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax__cont__UniformGrid_h
#define __dax__cont__UniformGrid_h

#include <dax/internal/GridStructures.h>

namespace dax {
namespace cont {

/// This class defines the topology of a uniform grid. A uniform grid is axis
/// aligned and has uniform spacing between grid points in every dimension. The
/// grid can be shifted and scaled in space by defining and origin and spacing.
///
class UniformGrid
{
public:
  UniformGrid() {
    this->SetOrigin(dax::make_Vector3(0.0, 0.0, 0.0));
    this->SetSpacing(dax::make_Vector3(1.0, 1.0, 1.0));
    this->SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(0, 0, 0));
  }

  /// The extent defines the minimum and maximum (inclusive) indices in each
  /// dimension.
  ///
  const dax::Extent3 &GetExtent() const { return this->GridStructure.Extent; }
  void SetExtent(const dax::Extent3 &extent) {
    this->GridStructure.Extent = extent;
  }
  void SetExtent(const dax::Id3 &min, const dax::Id3 &max) {
    this->GridStructure.Extent.Min = min;
    this->GridStructure.Extent.Max = max;
  }

  /// The origin is the location in space of the point at grid position
  /// (0, 0, 0).  This position may or may not actually be in the extent.
  ///
  const dax::Vector3 &GetOrigin() const { return this->GridStructure.Origin; }
  void SetOrigin(const dax::Vector3 &coords) {
    this->GridStructure.Origin = coords;
  }

  /// The spacing is the distance between grid points. Each component in the
  /// vector refers to a spacing along the associated axis, which can vary.
  ///
  const dax::Vector3 &GetSpacing() const { return this->GridStructure.Spacing; }
  void SetSpacing(const dax::Vector3 &distances) {
    this->GridStructure.Spacing = distances;
  }

  // Helper functions

  /// Get the number of points.
  ///
  dax::Id GetNumberOfPoints() const {
    return dax::internal::numberOfPoints(this->GridStructure);
  }

  /// Get the number of cells.
  ///
  dax::Id GetNumberOfCells() const {
    return dax::internal::numberOfCells(this->GridStructure);
  }

  /// Used internally to get a structure that can be passed to the execution
  /// environment.
  ///
  const dax::internal::StructureUniformGrid &GetStructureForExecution() const {
    return this->GridStructure;
  }

private:
  dax::internal::StructureUniformGrid GridStructure;
};

}
}

#endif //__dax__cont__UniformGrid_h
