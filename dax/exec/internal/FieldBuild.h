/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_internal_FieldBuild_h
#define __dax_exec_internal_FieldBuild_h

#include <dax/exec/Field.h>

#include <dax/internal/DataArray.h>
#include <dax/internal/GridStructures.h>

namespace dax { namespace exec { namespace internal {

DAX_EXEC_CONT_EXPORT dax::exec::FieldCoordinates fieldCoordinatesBuild(
    const dax::internal::StructureUniformGrid &)
{
  dax::internal::DataArray<dax::Vector3> dummyArray;
  dax::exec::FieldCoordinates field(dummyArray);
  return field;
}

}}}

#endif //__dax_exec_internal_FieldBuild_h
