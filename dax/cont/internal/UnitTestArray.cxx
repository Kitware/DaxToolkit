
/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/Types.h>
#include <dax/cont/Array.h>


#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace
{
const dax::Id ARRAY_SIZE = 10;

void test_assert(bool valid, const std::string& msg="")
{
  if(!valid)
    {
    std::stringstream error;
    error << __FILE__ << ":" << __LINE__ << std::endl;
    error << msg << std::endl;
    throw error.str();
    }
}

template <typename T>
T testValue()
  { return T(); }

template <>
dax::Id testValue()
  { return 4; }

template <>
dax::Scalar testValue()
  { return 1.61803399; }

template <>
dax::Vector3 testValue()
  { dax::Vector3 v = {0,1,2}; return v; }

template <>
dax::Vector4 testValue()
  { dax::Vector4 v = {3,2,1,0}; return v; }

template <typename T>
std::string testValueName(T)
  { return "unkown"; }

template <>
std::string testValueName(dax::Id)
  { return "dax::Id"; }

template <>
std::string testValueName(dax::Scalar)
  { return "dax::Scalar"; }

template <>
std::string testValueName(dax::Vector3)
  { return "dax::Vector3"; }

template <>
std::string testValueName(dax::Vector4)
  { return "dax::Vector4"; }
}


template<typename T>
void assignmentTest()
{
  dax::cont::Array<T> a1(ARRAY_SIZE);

  dax::cont::Array<T> a2(a1);
  dax::cont::Array<T> a3 = a1;
  test_assert(a2.size()==a1.size(), "copy constuctor failed");
  test_assert(a3.size()==a1.size(), "assignment operator failed");

  std::vector<T> v(a1.size());
  dax::cont::Array<T> vresult(v);
  test_assert(vresult.size()==a1.size(), "std::vector constructor failed");
}

template<typename T>
void resizeTest()
  {
  dax::cont::Array<T> a1(ARRAY_SIZE);
  a1.resize(ARRAY_SIZE*2);
  test_assert(a1.size()==ARRAY_SIZE*2, "resize failed");
  test_assert(a1.capacity()>=a1.size(), "capacity is incorrect");

  //now test that the resize on reduction
  //doesn't overwrite existing items
  a1.resize(2,testValue<T>());
  test_assert(a1.size()==2, "resize failed");
  test_assert(a1[0]!=testValue<T>(), "resize reduction shouldn't modify values");
  test_assert(a1[0]==a1[a1.size()-1], "start and end values should be identical");

  //now test resize on extension
  //properly uses the correct value
  a1.resize(4,testValue<T>());
  test_assert(a1.size()==4, "resize failed");
  test_assert(a1[2]==testValue<T>(), "values assigned at resize be at index 2");
  test_assert(a1[0]!=a1[2], "a[0] shouldn't equal a[2]");
  }

template<typename T>
void reserveTest()
  {
  dax::cont::Array<T> a1;
  test_assert(a1.capacity()>=a1.size(), "capacity is incorrect");
  std::size_t oldSize = a1.size();

  a1.reserve(ARRAY_SIZE);
  test_assert(a1.capacity()>=a1.size(), "capacity is incorrect");
  test_assert(a1.size()==oldSize, "reserve shouldn't modify size");
  }

template<typename T>
void pushBackTest()
  {
  dax::cont::Array<T> a1;
  a1.push_back(testValue<T>());
  test_assert(a1.size()==1, "size is wrong after push_back");
  test_assert(a1.capacity()>=1, "capacity is wrong after push_back");
  }

template<typename T>
void iteratorTest()
  {
  dax::cont::Array<T> a1;

  for(size_t i=0; i<5; ++i)
    {
    a1.push_back(testValue<T>());
    }

  typedef typename dax::cont::Array<T>::iterator Iterator;
  typedef typename dax::cont::Array<T>::const_iterator CIterator;

  std::size_t idx=0;
  for(Iterator it = a1.begin(); it != a1.end(); ++it)
    {
    test_assert( *it == a1[idx++], "iterator not returning same value as [] operator");
    }

  idx=0;
  for(CIterator it = a1.begin(); it != a1.end(); ++it)
    {
    test_assert( *it == a1[idx++], "const_iterator not returning same value as [] operator");
    }

  }

template<typename T>
void TestArray()
{
  std::cout << "Testing Array with data type " << testValueName(T()) << std::endl;
  assignmentTest<T>();
  resizeTest<T>();
  reserveTest<T>();
  pushBackTest<T>();
  iteratorTest<T>();
}

int UnitTestArray(int, char *[])
{
  try
    {
    TestArray<dax::Id>();
    TestArray<dax::Scalar>();
    TestArray<dax::Vector3>();
    TestArray<dax::Vector4>();
    }
  catch (std::string error)
    {
    std::cout
        << "Encountered error: " << std::endl
        << error << std::endl;
    return 1;
    }

  return 0;
}
