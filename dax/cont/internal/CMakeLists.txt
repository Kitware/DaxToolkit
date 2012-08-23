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
  DeviceAdapterAlgorithm.h
  DeviceAdapterAlgorithmSerial.h
  DeviceAdapterError.h
  DeviceAdapterTag.h
  DeviceAdapterTagSerial.h
  ScheduleGenerateTopology.h
  ScheduleMapAdapter.h
  Testing.h
  TestingDeviceAdapter.h
  TestingGridGenerator.h
  )

dax_declare_headers(${headers})

# These source files are actually compiled in the parent directory.
# They are in a seperate directory to highlight which objects are
# internal and which are part of the external interface.
#add_custom_target(daxContInternal ALL DEPENDS daxCont)

add_subdirectory(Testing)
