#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
set (headers
  CellTypes.h
  DataArray.h
  DataSet.h)

set(DAX_CORE_COMMON_HEADERS
  ${headers}
  CACHE INTERNAL
  "Headers in Core/Common")
