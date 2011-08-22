#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
set (headers
  DataArrayIrregular.h
  DataArrayStructuredConnectivity.h
  DataArrayStructuredPoints.h
  DataArrayTraits.h
  Macros.h
  Work.h)

set(DAX_CORE_EXECUTION_HEADERS
  ${headers}
  CACHE INTERNAL
  "Headers in Core/Execution")
