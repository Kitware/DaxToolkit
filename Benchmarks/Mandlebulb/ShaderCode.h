//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_Benchmarks_Mandlebulb_ShaderCode_h
#define __dax_Benchmarks_Mandlebulb_ShaderCode_h

namespace mandle
{
static std::string const& make_fragment_shader_code()
{
  static std::string const data =
"#version 120\n"
"varying vec4 vert_pos;"
"varying vec3 normal;"
"uniform float light_xpos;"
"void main()"
"{"
"  vec3 frag_to_light, light_pos, n;"
"  float brightness;"
"  /*move the light around the x axis*/"
"  light_pos = vec3(light_xpos,-1.0,1.0);"
"  frag_to_light = normalize(light_pos + vec3(vert_pos));"
"  n = normalize(normal);"
"  brightness = max(0.0, dot(n, frag_to_light));"
"  gl_FragColor = brightness * vec4(1,1,1,1) * gl_Color;"
"}";
  return data;
}


static std::string const& make_vertex_shader_code()
{
  static std::string const data =
"#version 120\n"
"varying vec4 vert_pos;"
"varying vec3 normal;"
"void main()"
"{"
"  normal = normalize(gl_NormalMatrix * gl_Normal);"
"  vert_pos = gl_ModelViewMatrix * gl_Vertex;"
"  gl_FrontColor =  gl_Color;"
"  gl_Position = ftransform();"
"}";

  return data;
}
}

#endif
