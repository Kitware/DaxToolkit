#ifndef DAXTYPES_H
#define DAXTYPES_H

namespace dax
{

typedef float Scalar;
typedef int Id;

/// Vector3 corresponds to a 3-tuple
struct Vector3 {
  dax::Scalar x; dax::Scalar y; dax::Scalar z;
};

struct Id3 {
  dax::Id x; dax::Id y; dax::Id z;
};

struct Extent3 {
  Id3 Min;
  Id3 Max;
  Extent3(Id3 min, Id3 max): Min(min), Max(max){}
};

inline Vector3 make_Vector3(float x, float y, float z)
  {
  Vector3 v={x,y,z};
  return v;
  }

inline Id3 make_Id3(int x, int y, int z)
  {
  Id3 v={x,y,z};
  return v;
  }

inline Scalar dot(const Vector3 &a, const Vector3 &b)
{
  return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

}

inline dax::Id3 operator+(const dax::Id3 &a,
                                             const dax::Id3 &b)
{
dax::Id3 result = { a.x + b.x, a.y + b.y, a.z + b.z };
return result;
}
inline dax::Id3 operator*(const dax::Id3 &a,
                                             const dax::Id3 &b)
{
dax::Id3 result = { a.x * b.x, a.y * b.y, a.z * b.z };
return result;
}
inline dax::Id3 operator-(const dax::Id3 &a,
                                             const dax::Id3 &b)
{
dax::Id3 result = { a.x - b.x, a.y - b.y, a.z - b.z };
return result;
}
inline dax::Id3 operator/(const dax::Id3 &a,
                                             const dax::Id3 &b)
{
dax::Id3 result = { a.x / b.x, a.y / b.y, a.z / b.z };
return result;
}
inline bool operator==(const dax::Id3 &a,
                                          const dax::Id3 &b)
{
return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}
inline bool operator!=(const dax::Id3 &a,
                                          const dax::Id3 &b)
{
return !(a == b);
}

inline dax::Vector3 operator+(const dax::Vector3 &a,
                                                 const dax::Vector3 &b)
{
dax::Vector3 result = { a.x + b.x, a.y + b.y, a.z + b.z };
return result;
}
inline dax::Vector3 operator*(const dax::Vector3 &a,
                                                 const dax::Vector3 &b)
{
dax::Vector3 result = { a.x * b.x, a.y * b.y, a.z * b.z };
return result;
}
inline dax::Vector3 operator-(const dax::Vector3 &a,
                                                 const dax::Vector3 &b)
{
dax::Vector3 result = { a.x - b.x, a.y - b.y, a.z - b.z };
return result;
}
inline dax::Vector3 operator/(const dax::Vector3 &a,
                                                 const dax::Vector3 &b)
{
dax::Vector3 result = { a.x / b.x, a.y / b.y, a.z / b.z };
return result;
}
inline bool operator==(const dax::Vector3 &a,
                                          const dax::Vector3 &b)
{
return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}
inline bool operator!=(const dax::Vector3 &a,
                                          const dax::Vector3 &b)
{
return !(a == b);
}

inline dax::Vector3 operator*(dax::Scalar a,
                                                 const dax::Vector3 &b)
{
dax::Vector3 result = { a * b.x, a * b.y, a * b.z };
return result;
}

inline dax::Vector3 operator*(const dax::Vector3 &a,
                                                 dax::Scalar &b)
{
dax::Vector3 result = { a.x * b, a.y * b, a.z * b };
return result;
}

inline dax::Vector3 operator*(dax::Id3 a,const dax::Vector3 &b)
{
dax::Vector3 result = { a.x * b.x, a.y * b.y, a.z * b.z };
return result;
}

inline dax::Vector3 operator*(const dax::Vector3 &a, dax::Id3 &b)
{
dax::Vector3 result = { a.x * b.x, a.y * b.y, a.z * b.z };
return result;
}

#endif // DAXTYPES_H
