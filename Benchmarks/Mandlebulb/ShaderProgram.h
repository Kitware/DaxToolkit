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
#ifndef __dax_Benchmarks_Mandlebulb_ShaderProgram_h
#define __dax_Benchmarks_Mandlebulb_ShaderProgram_h

#include <algorithm>
#include <string>
#include <fstream>
#include <vector>

namespace mandle
{
  static bool bind_shader(const std::string& shader_contents, GLenum type,
                     std::vector<GLuint>& storage)
  {
      GLuint shader_id = glCreateShader( type );
      if(shader_id > 0)
        {
        const char *c_str = shader_contents.c_str();
        glShaderSource(shader_id,
                       1,
                       reinterpret_cast<const GLchar**>(&c_str),
                       NULL);

        //compile and get if compilation worked
        GLint compiled_status;
        glCompileShader(shader_id);
        glGetShaderiv(shader_id, GL_COMPILE_STATUS, &compiled_status);

        if(compiled_status == GL_TRUE)
          {
          storage.push_back(shader_id);
          return true;
          }
        else
          {
          GLint errorMsgLen;
          glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &errorMsgLen);
          char* errorMsg = new char[errorMsgLen + 1];
          glGetShaderInfoLog(shader_id, errorMsgLen, NULL, errorMsg);
          std::cerr << errorMsg << std::endl;
          delete[] errorMsg;
          glDeleteShader(shader_id);
          }
        }
    return false;
  }


  class ShaderProgram
  {
  public:
    //construct a shader based on the contents of a string
    ShaderProgram():
      ProgramId(0),
      ShaderIds()
    {
    }

    ~ShaderProgram()
    {
      //release each shader
      if(this->ProgramId != 0)
        {
        this->release_shaders();
        }
    }

    bool add_vert_shader(const std::string& code)
      { return bind_shader(code,GL_VERTEX_SHADER,ShaderIds); }
    bool add_frag_shader(const std::string& code)
      { return bind_shader(code,GL_FRAGMENT_SHADER,ShaderIds); }

    //will build the program
    bool build();

    //get the gl program id
    GLuint program_id() const { return this->ProgramId; }

  private:
    void release_shaders();

    GLuint ProgramId;
    std::vector<GLuint> ShaderIds;
  };


inline bool ShaderProgram::build()
{
  typedef std::vector<GLuint>::const_iterator c_it;
  if(this->ProgramId != 0)
    {
    this->release_shaders();
    }

  this->ProgramId = glCreateProgram();
  for(c_it i = this->ShaderIds.begin(); i != this->ShaderIds.end(); ++i)
    { glAttachShader(this->ProgramId, *i); }

  glLinkProgram(this->ProgramId);

  for(c_it i = this->ShaderIds.begin(); i != this->ShaderIds.end(); ++i)
    { glDetachShader(this->ProgramId, *i); }

  return true;
}

//presumes that ProgramId isn't 0 and we have bound some shaders already
inline void ShaderProgram::release_shaders()
{
  typedef std::vector<GLuint>::const_iterator c_it;
  //clear the shaders
  for(c_it i = this->ShaderIds.begin(); i != this->ShaderIds.end(); ++i)
    { glDeleteShader(*i); }
  this->ShaderIds.clear();

  //clear the program
  glDeleteProgram(this->ProgramId);
  this->ProgramId=0;
}

}

#endif
