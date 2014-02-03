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
set(headers
  ArgumentsParser.h
  CoolWarmColorMap.h
  Mandlebulb.h
  ShaderProgram.h
  ShaderCode.h
  Window.h
  Worklets.h
  )

set(sources
  ArgumentsParser.cxx
  Mandlebulb.cxx
  Window.cxx
  main.cxx
  )

#-----------------------------------------------------------------------------
set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY TRUE)

#-----------------------------------------------------------------------------
add_executable(MandlebulbSerial ${headers} ${sources})
set_dax_device_adapter(MandlebulbSerial DAX_DEVICE_ADAPTER_SERIAL)
add_test(MandlebulbSerial ${EXECUTABLE_OUTPUT_PATH}/MandlebulbSerial --time=5000 --auto-play)
target_link_libraries(MandlebulbSerial ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})

#-----------------------------------------------------------------------------
if (DAX_ENABLE_OPENMP)
  add_executable(MandlebulbOpenMP ${headers} ${sources})
  set_dax_device_adapter(MandlebulbOpenMP DAX_DEVICE_ADAPTER_OPENMP)
  add_test(MandlebulbOpenMP ${EXECUTABLE_OUTPUT_PATH}/MandlebulbOpenMP --time=5000 --auto-play)
  target_link_libraries(MandlebulbOpenMP ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})
endif (DAX_ENABLE_OPENMP)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_TBB)
  add_executable(MandlebulbTBB ${headers} ${sources})
  set_dax_device_adapter(MandlebulbTBB DAX_DEVICE_ADAPTER_TBB)
  add_test(MandlebulbTBB ${EXECUTABLE_OUTPUT_PATH}/MandlebulbTBB --time=5000 --auto-play)
  target_link_libraries(MandlebulbTBB ${OPENGL_LIBRARIES}
                        ${GLUT_LIBRARIES} ${GLEW_LIBRARIES} ${TBB_LIBRARIES})
endif (DAX_ENABLE_TBB)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_CUDA)
  dax_disable_troublesome_thrust_warnings()
  set(cuda_sources ${sources})
  list(REMOVE_ITEM cuda_sources Mandlebulb.cxx)
  list(REMOVE_ITEM cuda_sources Window.cxx )
  list(REMOVE_ITEM cuda_sources main.cxx )
  list(APPEND cuda_sources Mandlebulb.cu Window.cu  main.cu)
  cuda_add_executable(MandlebulbCuda ${headers} ${cuda_sources})
  set_dax_device_adapter(MandlebulbCuda DAX_DEVICE_ADAPTER_CUDA)
  add_test(MandlebulbCuda ${EXECUTABLE_OUTPUT_PATH}/MandlebulbCuda --time=5000 --auto-play)
  target_link_libraries(MandlebulbCuda
                        ${OPENGL_LIBRARIES}
                        ${GLUT_LIBRARIES}
                        ${GLEW_LIBRARIES})
endif (DAX_ENABLE_CUDA)
