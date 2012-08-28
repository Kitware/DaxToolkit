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

#-----------------------------------------------------------------------------
#Include VTK so we can benchmark agianst it
option(DAX_ENABLE_VTK_TESTING FALSE)
if(DAX_ENABLE_VTK_TESTING)
  find_package(VTK REQUIRED)
endif()

#-----------------------------------------------------------------------------
if(DAX_ENABLE_VTK_TESTING)
  include(${VTK_USE_FILE})
  add_executable(testMC TestMarchingCubes.cxx)
  target_link_libraries(testMC vtkCommonDataModel vtkIOGeometry)
endif()
