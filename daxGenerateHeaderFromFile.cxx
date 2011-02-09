/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// Simple program to generate a C++ header file from a file.
#include "daxSystemIncludes.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <fstream>
#include <iterator>
#include <vector>
#include <string>
using namespace std;

namespace
{
  /**
   * Return file name of a full filename (i.e. file name without path).
   */
  std::string GetFilenameName(const std::string& filename)
    {
#if defined(_WIN32)
    std::string::size_type slash_pos = filename.find_last_of("/\\");
#else
    std::string::size_type slash_pos = filename.find_last_of("/");
#endif
    if(slash_pos != std::string::npos)
      {
      return filename.substr(slash_pos + 1);
      }
    else
      {
      return filename;
      }
    }

  /**
   * Return file name without extension of a full filename (i.e. without path).
   * Warning: it considers the longest extension (for example: .tar.gz)
   */
  std::string GetFilenameWithoutExtension(const std::string& filename)
    {
    std::string name = GetFilenameName(filename);
    std::string::size_type dot_pos = name.find(".");
    if(dot_pos != std::string::npos)
      {
      return name.substr(0, dot_pos);
      }
    else
      {
      return name;
      }
    }


  /**
   * Return file name without extension of a full filename (i.e. without path).
   * Warning: it considers the last extension (for example: removes .gz
   * from .tar.gz)
   */
  std::string GetFilenameWithoutLastExtension(const std::string& filename)
    {
    std::string name = GetFilenameName(filename);
    std::string::size_type dot_pos = name.rfind(".");
    if(dot_pos != std::string::npos)
      {
      return name.substr(0, dot_pos);
      }
    else
      {
      return name;
      }
    }
}


int main(int argc, char**argv)
{
  po::options_description desc("Allowed options");
  desc.add_options()
    ("input-file", po::value< vector<string> >(), "Input files (required)")
    ("output-file", po::value< vector<string > >(), "Output files (required)")
    ("help", "Generate this help message");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help") != 0 ||
    vm.count("input-file") == 0 ||
    vm.count("output-file") == 0)
    {
    cout << desc << endl;
    return 1;
    }
  if (vm.count("input-file") != vm.count("output-file"))
    {
    cout << "Input and output files must match up.";
    cout << desc << endl;
    return 1;
    }

  vector<string> input_files = vm["input-file"].as< vector<string> >();
  vector<string> output_files = vm["output-file"].as< vector<string> >();

  for (size_t cc=0; cc < input_files.size(); cc++)
    {
    ifstream infile(input_files[cc].c_str());
    ofstream outfile(output_files[cc].c_str());
    string name = GetFilenameWithoutExtension(input_files[cc]);
    if (infile.is_open() && outfile.is_open())
      {
      outfile <<
        "/*=========================================================================\n"
        "\n"
        "  This software is distributed WITHOUT ANY WARRANTY; without even\n"
        "  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR\n"
        "  PURPOSE.  See the above copyright notice for more information.\n"
        "\n"
        "=========================================================================*/\n"
        "// **** Generated from :\n// " << input_files[cc].c_str() << "\n"
        "\n"
        "static const char* daxHeaderString_" << name.c_str() << " =\n\"";
      while (infile.good())
        {
        char ch;
        infile.get(ch);
        if (infile.good())
          {
          if ( ch == '\n' )
            {
            outfile << "\\n\"" << endl << "\"";
            }
          else if ( ch == '\\' )
            {
            outfile << "\\\\";
            }
          else if ( ch == '\"' )
            {
            outfile << "\\\"";
            }
          else if ( ch != '\r' )
            {
            outfile << static_cast<unsigned char>(ch);
            }
          }
        }
      outfile << "\";\n";
      }
    infile.close();
    outfile.close();
    }
  return 0;
}
