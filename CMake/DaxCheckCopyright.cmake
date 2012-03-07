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

## This CMake script checks source files for the appropriate Dax copyright
## statement, which is stored in Dax_SOURCE_DIR/CMake/DaxCopyrightStatement.txt.
## To run this script, execute CMake as follows:
##
## cmake -DDax_SOURCE_DIR=<Dax_SOURCE_DIR> -P <Dax_SOURCE_DIR>/CMake/DaxCheckCopyright.cmake
##
## or more simply
##
## cmake -P <path>/DaxCheckCopyright.cmake
##

cmake_minimum_required(VERSION 2.8)

set(FILES_TO_CHECK
  *.txt
  *.cmake
  *.h
  *.h.in
  *.cxx
  *.cu
  *.worklet
  )

set(EXCEPTIONS
  LICENSE.txt
  )

if (NOT Dax_SOURCE_DIR)
  get_filename_component(Dax_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. REALPATH)
endif (NOT Dax_SOURCE_DIR)

set(copyright_file ${Dax_SOURCE_DIR}/CMake/DaxCopyrightStatement.txt)

if (NOT EXISTS ${copyright_file})
  set(copyright_file ${CMAKE_CURRENT_SOURCE_DIR}/DaxCopyrightStatement.txt)
endif (NOT EXISTS ${copyright_file})

if (NOT EXISTS ${copyright_file})
  message(SEND_ERROR "Cannot find DaxCopyrightStatement.txt.")
endif (NOT EXISTS ${copyright_file})

# Gets the current year (if possible).
function (get_year var)
  if (UNIX)
    execute_process(COMMAND "date" "+%Y" OUTPUT_VARIABLE result)
    string(REGEX REPLACE "(....).*" "\\1" result "${result}")
  elseif (WIN32)
    execute_process(COMMAND "date" "/T" OUTPUT_VARIABLE result)
    string(REGEX REPLACE ".*../../(....).*" "\\1" result "${result}")
  else (UNIX)
    message("Don't know how to get date.")
    set(result "20XX")
  endif (UNIX)
  set(${var} "${result}" PARENT_SCOPE)
endfunction (get_year)

set(copyright_file_year 2011)
get_year(current_year)

# Escapes ';' characters (list delimiters) and splits the given string into
# a list of its lines without newlines.
function (list_of_lines var string)
  string(REGEX REPLACE ";" "\\\\;" conditioned_string "${string}")
  string(REGEX REPLACE "\n" ";" conditioned_string "${conditioned_string}")
  set(${var} "${conditioned_string}" PARENT_SCOPE)
endfunction (list_of_lines)

# Read in copyright statement file.
file(READ ${copyright_file} COPYRIGHT_STATEMENT)

# Remove trailing whitespace and ending lines.  They are sometimes hard to
# see or remove in editors.
string(REGEX REPLACE "[ \t]*\n" "\n" COPYRIGHT_STATEMENT "${COPYRIGHT_STATEMENT}")
string(REGEX REPLACE "\n+$" "" COPYRIGHT_STATEMENT "${COPYRIGHT_STATEMENT}")

# Get a list of lines in the copyright statement.
list_of_lines(COPYRIGHT_LINE_LIST "${COPYRIGHT_STATEMENT}")

# Comment regular expression characters that we want to match literally.
string(REPLACE "." "\\." COPYRIGHT_LINE_LIST "${COPYRIGHT_LINE_LIST}")
string(REPLACE "(" "\\(" COPYRIGHT_LINE_LIST "${COPYRIGHT_LINE_LIST}")
string(REPLACE ")" "\\)" COPYRIGHT_LINE_LIST "${COPYRIGHT_LINE_LIST}")

# Introduce regular expression for years we want to be generic.
string(REPLACE
  "${copyright_file_year}"
  "20[0-9][0-9]"
  COPYRIGHT_LINE_LIST
  "${COPYRIGHT_LINE_LIST}"
  )

# Replace year in COPYRIGHT_STATEMENT with current year.
string(REPLACE
  "${copyright_file_year}"
  "${current_year}"
  COPYRIGHT_STATEMENT
  "${COPYRIGHT_STATEMENT}"
  )

# Print an error concerning the missing copyright in the given file.
function(missing_copyright filename comment_prefix)
  message("${filename} does not have the appropriate copyright statement:\n")

  # Condition the copyright statement
  string(REPLACE
    "\n"
    "\n${comment_prefix}  "
    comment_copyright
    "${COPYRIGHT_STATEMENT}"
    )
  set(comment_copyright "${comment_prefix}  ${comment_copyright}")
  string(REPLACE
    "\n${comment_prefix}  \n"
    "\n${comment_prefix}\n"
    comment_copyright
    "${comment_copyright}"
    )

  message("${comment_prefix}=============================================================================")
  message("${comment_prefix}")
  message("${comment_copyright}")
  message("${comment_prefix}")
  message("${comment_prefix}=============================================================================\n")
  message(SEND_ERROR
    "Please add the previous statement to the beginning of ${filename}"
    )
endfunction(missing_copyright)

# Get an appropriate beginning line comment for the given filename.
function(get_comment_prefix var filename)
  get_filename_component(base "${filename}" NAME_WE)
  get_filename_component(extension "${filename}" EXT)
  if (extension STREQUAL ".cmake")
    set(${var} "##" PARENT_SCOPE)
  elseif (base STREQUAL "CMakeLists" AND extension STREQUAL ".txt")
    set(${var} "##" PARENT_SCOPE)
  elseif (extension STREQUAL ".txt")
    set(${var} "" PARENT_SCOPE)
  elseif (extension STREQUAL ".h" OR extension STREQUAL ".h.in" OR extension STREQUAL ".cxx" OR extension STREQUAL ".cu")
    set(${var} "//" PARENT_SCOPE)
  elseif (extension STREQUAL ".worklet")
    set(${var} "//" PARENT_SCOPE)
  else (extension STREQUAL ".cmake")
    message(SEND_ERROR "Could not identify file type of ${filename}.")
  endif (extension STREQUAL ".cmake")
endfunction(get_comment_prefix)

# Check the given file for the appropriate copyright statement.
function(check_copyright filename)
  get_comment_prefix(comment_prefix "${filename}")

  # Read in the first 2000 characters of the file and split into lines.
  # This is roughly equivalent to the file STRINGS command except that we
  # also escape semicolons (list separators) in the input, which the file
  # STRINGS command does not currently do.
  file(READ "${filename}" header_contents LIMIT 2000)
  list_of_lines(header_lines "${header_contents}")

  # Check each copyright line.
  foreach (copyright_line IN LISTS COPYRIGHT_LINE_LIST)
    set(match)
    # My original algorithm tried to check the order by removing items from
    # header_lines as they were encountered.  Unfortunately, CMake 2.8's
    # list REMOVE_AT command removed the escaping on the ; in one of the
    # header_line's items and cause the compare to fail.
    foreach (header_line IN LISTS header_lines)
      if (copyright_line)
	string(REGEX MATCH
	  "^${comment_prefix}[ \t]*${copyright_line}[ \t]*$"
	  match
	  "${header_line}"
	  )
      else (copyright_line)
	if (NOT header_line)
	  set(match TRUE)
	endif (NOT header_line)
      endif (copyright_line)
      if (match)
	break()
      endif (match)
    endforeach (header_line)
    if (NOT match)
      message("Could not find match for `${copyright_line}'")
      missing_copyright("${filename}" "${comment_prefix}")
      break()
    endif (NOT match)
  endforeach (copyright_line)
endfunction(check_copyright)

foreach (glob_expression ${FILES_TO_CHECK})
  file(GLOB_RECURSE file_list
    RELATIVE "${Dax_SOURCE_DIR}"
    "${Dax_SOURCE_DIR}/${glob_expression}"
    )
  list(REMOVE_ITEM file_list ${EXCEPTIONS})
  foreach (file ${file_list})
    message("Checking ${file}")
    check_copyright("${Dax_SOURCE_DIR}/${file}")
  endforeach (file)
endforeach (glob_expression)
