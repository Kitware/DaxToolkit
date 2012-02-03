/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <dax/exec/Derivative.h>

#include <dax/exec/WorkMapCell.h>
#include <dax/exec/WorkMapField.h>
#include <dax/exec/internal/ErrorHandler.h>
#include <dax/internal/GridTopologys.h>

#include <dax/internal/Testing.h>

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

/// An (invalid) error handler to pass to work constructors.
dax::exec::internal::ErrorHandler ErrorHandler
  = dax::exec::internal::ErrorHandler(dax::internal::DataArray<char>());

} // Anonymous namespace

const dax::Id bufferSize = 1024*1024;
static dax::Scalar fieldBuffer[bufferSize];

template<class CellType>
static dax::exec::FieldPoint<dax::Scalar> CreatePointField(
    dax::exec::WorkMapField<CellType> work,
    const dax::exec::FieldCoordinates &coordField,
    const LinearField &fieldValues,
    dax::Id numPoints)
{
  DAX_TEST_ASSERT(bufferSize >= numPoints,
                  "Internal test error.  Buffer not large enough");

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
        DAX_TEST_ASSERT(computedDerivative == fieldValues.Gradient,
                        "Bad derivative");
        }
      }
    }
}

static void TestDerivativeVoxel(
    const dax::internal::TopologyUniform &gridstruct,
    const LinearField &fieldValues)
{
  dax::exec::WorkMapField<dax::exec::CellVoxel> workField(gridstruct,
                                                          ErrorHandler);
  dax::exec::FieldCoordinates coordField
      = dax::exec::FieldCoordinates(
          dax::internal::DataArray<dax::Vector3>());
  dax::Id numPoints = dax::internal::numberOfPoints(gridstruct);
  dax::exec::FieldPoint<dax::Scalar> scalarField
      = CreatePointField(workField, coordField, fieldValues, numPoints);

  dax::exec::WorkMapCell<dax::exec::CellVoxel> workCell(gridstruct,
                                                        ErrorHandler);
  dax::Id numCells = dax::internal::numberOfCells(gridstruct);
  for (dax::Id cellIndex = 0; cellIndex < numCells; cellIndex++)
    {
    workCell.SetCellIndex(cellIndex);
    TestDerivativeCell(workCell, coordField, scalarField, fieldValues);
    }
}

static void TestDerivativeVoxel()
{
  dax::internal::TopologyUniform gridstruct;
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

static void TestDerivatives()
{
  TestDerivativeVoxel();
}

int UnitTestDerivative(int, char *[])
{
  return dax::internal::Testing::Run(TestDerivatives);
}
