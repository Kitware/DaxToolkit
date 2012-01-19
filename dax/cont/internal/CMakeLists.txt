#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  ArrayContainerExecutionCPU.h
  ExecutionPackageField.h
  ExecutionPackageGrid.h
  IteratorContainer.h
  IteratorPolymorphic.h
  )

dax_declare_headers(${headers})

# These source files are actually compiled in the parent directory.
# They are in a seperate directory to highlight which objects are
# internal and which are part of the external interface.
#add_custom_target(daxContInternal ALL DEPENDS daxCont)

set(unit_tests
  UnitTestArrayContainerExecutionCPU.cxx
  UnitTestExecutionPackage.cxx
  UnitTestIteratorPolymorphic.cxx
  )
dax_unit_tests(SOURCES ${unit_tests})
