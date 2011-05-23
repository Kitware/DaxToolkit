cmake_minimum_required(VERSION 2.8)
project (Functors)

#set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})
find_package(CUDA REQUIRED)

FIND_PATH(GOOGLE_CTEMPLATE_INCLUDE_DIR google/template.h /usr/include /usr/local/include)
FIND_LIBRARY(GOOGLE_CTEMPLATE_THREAD_LIB ctemplate /usr/lib /usr/loca/lib)


# List of Boost features used:
# * Smart Ptr
# * xpressive
# * program options
find_package(Boost REQUIRED
  COMPONENTS program_options # program_options is only needed for the testing
                             # executables
  )

include_directories(${Boost_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${GOOGLE_CTEMPLATE_INCLUDE_DIR}
  )

add_executable(daxGenerateHeaderFromFile
  daxGenerateHeaderFromFile.cxx)
target_link_libraries(daxGenerateHeaderFromFile
  ${Boost_LIBRARIES})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/daxOptions.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/daxOptions.h
  @ONLY)

function (encode_to_headers out_variable)
  set (all_headers)
  foreach(file ${ARGN})
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    get_filename_component(filename ${file} NAME)
    set(res ${CMAKE_CURRENT_BINARY_DIR}/${filename}.h)
    list(APPEND all_headers ${res})
    add_custom_command(
      OUTPUT ${res}
      DEPENDS ${src} daxGenerateHeaderFromFile
      COMMAND daxGenerateHeaderFromFile
      ARGS  --input-file ${src} --output-file ${res}
      )
  endforeach()
  set (${out_variable} ${all_headers} PARENT_SCOPE)
endfunction()

set (functor_modules
)

set (core_kernels
  Resources/Kernel.tmpl
  Resources/KernelGetArray.tmpl
)

set (generated_headers_core)
encode_to_headers(generated_headers_core
  ${core_kernels}
  )
set (generated_headers)
encode_to_headers(generated_headers
  ${functor_modules}
  )
#message ("Headers: ${generated_headers}")

CUDA_ADD_LIBRARY(DaxCuda
  Resources/DaxDataTypes.cu
  Resources/DaxAPI.cu
  SHARED)

# daxFramework is the core library
add_library(daxFramework SHARED
  daxExecutive.cxx
  daxExecutive.h
  daxModule.cxx
  daxModule.h
  daxObject.cxx
  daxObject.h
  daxPort.cxx
  daxPort.h
  ${generated_headers_core}
)

target_link_libraries(daxFramework
  ${Boost_LIBRARIES}
  ${GOOGLE_CTEMPLATE_THREAD_LIB}
  DaxCuda
  )

foreach (file ${generated_headers_core})
  set_source_files_properties(daxObject.h
    PROPERTIES OBJECT_DEPENDS ${file})
endforeach()

# Add executables, mostly testing samples
add_executable(daxTest
  daxTest.cxx
  ${generated_headers}
  )

foreach (file ${generated_headers})
  set_source_files_properties(daxTest.cxx
    PROPERTIES OBJECT_DEPENDS ${file})
endforeach()

target_link_libraries(daxTest
  daxFramework
  )
