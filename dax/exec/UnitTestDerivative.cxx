/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <dax/exec/Derivative.h>

#include <dax/exec/WorkMapCell.h>
#include <dax/exec/WorkMapField.h>
#include <dax/internal/GridStructures.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <vector>


#define TEST_FAIL(msg)                                  \
  {                                                     \
    std::stringstream error;                            \
    error << __FILE__ << ":" << __LINE__ << std::endl;  \
    error msg;                                          \
    throw error.str();                                  \
  }

namespace
{
/// Simple structure describing a linear field.  Has a convienience class
/// for getting values.
struct LinearField {
  dax::Vector3 Gradient;
  dax::Scalar OriginValue;

  dax::Scalar GetValue(dax::Vector3 coordinates) const {
    return dax::dot(coordinates, this->Gradient) + this->OriginValue;
  }
};
}

const dax::Id bufferSize = 1024*1024;
static dax::Scalar fieldBuffer[bufferSize];

template<class CellType>
static dax::exec::FieldPoint<dax::Scalar> CreatePointField(
    dax::exec::WorkMapField<CellType> work,
    const dax::exec::FieldCoordinates &coordField,
    const LinearField &fieldValues,
    dax::Id numPoints)
{
  if (bufferSize < numPoints)
    {
    TEST_FAIL(<< "Internal test error.  Buffer not large enough");
    }

  // Create field.
  dax::internal::DataArray<dax::Scalar> fieldData(fieldBuffer, numPoints);
  dax::exec::FieldPoint<dax::Scalar> field(fieldData);

  // Fill field.
  for (dax::Id pointIndex = 0; pointIndex < numPoints; pointIndex++)
    {
    work.SetIndex(pointIndex);
    dax::Vector3 coordinates = work.GetFieldValue(coordField);
    dax::Scalar fieldValue = fieldValues.GetValue(coordinates);
    work.SetFieldValue(field, fieldValue);
    }

  return field;
}

template<class CellType>
static void TestDerivativeCell(
    const dax::exec::WorkMapCell<CellType> &work,
    const dax::exec::FieldCoordinates &coordField,
    const dax::exec::FieldPoint<dax::Scalar> &scalarField,
    const LinearField &fieldValues)
{
  CellType cell = work.GetCell();

  dax::Vector3 pcoords;
  for (pcoords[2] = 0.0; pcoords[2] <= 1.0; pcoords[2] += 0.25)
    {
    for (pcoords[1] = 0.0; pcoords[1] <= 1.0; pcoords[1] += 0.25)
      {
      for (pcoords[0] = 0.0; pcoords[0] <= 1.0; pcoords[0] += 0.25)
        {
        dax::Vector3 computedDerivative
            = dax::exec::cellDerivative(work,
                                        cell,
                                        pcoords,
                                        coordField,
                                        scalarField);
        if (computedDerivative != fieldValues.Gradient)
          {
          TEST_FAIL(<< "Bad derivative");
          }
        }
      }
    }
}

static void TestDerivativeVoxel(
    const dax::internal::StructureUniformGrid &gridstruct,
    const LinearField &fieldValues)
{
  dax::exec::WorkMapField<dax::exec::CellVoxel> workField(gridstruct, 0);
  dax::exec::FieldCoordinates coordField
      = dax::exec::FieldCoordinates(
          dax::internal::DataArray<dax::Vector3>());
  dax::Id numPoints = dax::internal::numberOfPoints(gridstruct);
  dax::exec::FieldPoint<dax::Scalar> scalarField
      = CreatePointField(workField, coordField, fieldValues, numPoints);

  dax::Id numCells = dax::internal::numberOfCells(gridstruct);
  for (dax::Id cellIndex = 0; cellIndex < numCells; cellIndex++)
    {
    dax::exec::WorkMapCell<dax::exec::CellVoxel> workCell(gridstruct,cellIndex);
    TestDerivativeCell(workCell, coordField, scalarField, fieldValues);
    }
}

static void TestDerivativeVoxel()
{
  dax::internal::StructureUniformGrid gridstruct;
  LinearField fieldValues;

  std::cout << "Very simple field." << std::endl;
  gridstruct.Origin = dax::make_Vector3(0.0, 0.0, 0.0);
  gridstruct.Spacing = dax::make_Vector3(1.0, 1.0, 1.0);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  fieldValues.Gradient = dax::make_Vector3(1.0, 1.0, 1.0);
  fieldValues.OriginValue = 0.0;
  TestDerivativeVoxel(gridstruct, fieldValues);

  std::cout << "Uneven spacing/gradient." << std::endl;
  gridstruct.Origin = dax::make_Vector3(1.0, -0.5, 13.0);
  gridstruct.Spacing = dax::make_Vector3(2.5, 6.25, 1.0);
  gridstruct.Extent.Min = dax::make_Id3(5, -2, -7);
  gridstruct.Extent.Max = dax::make_Id3(20, 4, 10);
  fieldValues.Gradient = dax::make_Vector3(0.25, 14.0, 11.125);
  fieldValues.OriginValue = -7.0;
  TestDerivativeVoxel(gridstruct, fieldValues);

  std::cout << "Negative gradient directions." << std::endl;
  gridstruct.Origin = dax::make_Vector3(-5.0, -5.0, -5.0);
  gridstruct.Spacing = dax::make_Vector3(1.0, 1.0, 1.0);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  fieldValues.Gradient = dax::make_Vector3(-11.125, -0.25, 14.0);
  fieldValues.OriginValue = 5.0;
  TestDerivativeVoxel(gridstruct, fieldValues);
}

int UnitTestDerivative(int, char *[])
{
  try
    {
    TestDerivativeVoxel();
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
