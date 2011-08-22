#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

# All files in the Interface/Execution kit are headers. So not sure what do with
# them. Eventually, we'll have install rules.

set (headers
  Cell.h
  Field.h
  Work.h
)

set (DAX_INTERFACE_EXECUTION_HEADERS
  ${headers}
  CACHE INTERNAL
  "Headers in Interface/Execution")
