if(CMAKE_COMPILER_IS_GNUCXX)

  include(CheckCXXCompilerFlag)

  # Standard warning flags we should always have
  set(CMAKE_CXX_FLAGS_WARN " -Wall")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CMAKE_CXX_FLAGS_WARN}")
  set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_WARN}")

  # Addtional warnings for GCC
  set(CMAKE_CXX_FLAGS_WARN_EXTRA "-Wno-long-long -ansi -Wcast-align -Wchar-subscripts -Wextra -Wpointer-arith -Wformat-security -Wshadow -Wunused-parameter -fno-common")
  # Set up the debug CXX_FLAGS for extra warnings
  option(DAX_EXTRA_COMPILER_WARNINGS "Add compiler flags to do stricter checking when building debug." OFF)
  if(DAX_EXTRA_COMPILER_WARNINGS)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CMAKE_CXX_FLAGS_WARN_EXTRA}")
    set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_WARN_EXTRA}")
  endif()

  #add in support for debugging Thrust when building in debug mode
  set(CMAKE_CXX_FLAGS_DEBUG_THRUST "-DTHRUST_DEBUG")
  option(DAX_DEBUG_THRUST "Add in support for thrust debugging" OFF)
  if(DAX_DEBUG_THRUST)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CMAKE_CXX_FLAGS_DEBUG_THRUST}")
    set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_DEBUG_THRUST}")
  endif()

endif()

