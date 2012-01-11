/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_internal_ExecutionPackageGrid_h
#define __dax_cont_internal_ExecutionPackageGrid_h

#include <dax/internal/ExportMacros.h>

namespace dax {
namespace cont {
namespace internal {

template<class GridT>
class ExecutionPackageGrid
{
public:
  typedef GridT ControlGridType;
  typedef typename GridT::StructureType ExecutionGridType;
  typedef typename GridT::CellType ExecutionCellType;

  ExecutionPackageGrid(const ControlGridType &grid)
    : GridStructure(grid.GridStructure) { }

  ExecutionPackageGrid(const ExecutionGridType &grid) : GridStructure(grid) { }

  const ExecutionGridType &GetExecutionObject() const {
    return this->GridStructure;
  }
private:
  ExecutionGridType GridStructure;
};

}
}
}

#endif //__dax_cont_internal_ExecutionPackageGrid_h
