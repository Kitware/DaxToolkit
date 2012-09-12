##=============================================================================
##
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2012 Sandia Corporation.
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##
##=============================================================================

set(headers
  ArrayContainerControlError.h
  ArrayManagerExecution.h
  ArrayManagerExecutionSerial.h
  ArrayManagerExecutionShareWithControl.h
  ArrayPortalShrink.h
  Bindings.h
  DeviceAdapterAlgorithm.h
  DeviceAdapterAlgorithmSerial.h
  DeviceAdapterError.h
  DeviceAdapterTag.h
  DeviceAdapterTagSerial.h
  FindBinding.h
  ${Dax_BINARY_DIR}/dax/cont/internal/Members.h
  ScheduleMapAdapter.h
  Testing.h
  TestingDeviceAdapter.h
  TestingGridGenerator.h
  )

dax_declare_headers(${headers})

#-----------------------------------------------------------------------------
set(DAX_MEMBERS_CONFIGURED_INCLUDE_PATH "dax/cont/internal")
set(DAX_MEMBERS_NAMESPACE "dax::cont::internal")
set(DAX_MEMBERS_NAMESPACE_NAME "dax_cont_internal")
set(DAX_MEMBERS_NAMESPACE_START "namespace dax { namespace cont { namespace internal {")
set(DAX_MEMBERS_NAMESPACE_END   "} } }")
set(DAX_MEMBERS_EXPORT "DAX_CONT_EXPORT")
configure_file(
  ${Dax_SOURCE_DIR}/dax/internal/Members.h.in
  ${Dax_BINARY_DIR}/dax/cont/internal/Members.h
  @ONLY
  )

# These source files are actually compiled in the parent directory.
# They are in a seperate directory to highlight which objects are
# internal and which are part of the external interface.
#add_custom_target(daxContInternal ALL DEPENDS daxCont)

set(unit_tests
  UnitTestArrayManagerExecutionShareWithControl.cxx
  UnitTestBindings.cxx
  UnitTestContTesting.cxx
  )
dax_unit_tests(SOURCES ${unit_tests})
