// © 2021 ETH Zurich, Mechanics and Materials Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ae108/assembly/Assembler.h"
#include "ae108/cmdline/CommandLineOptionParser.h"
#include "ae108/cpppetsc/GeneralizedMeshBoundaryCondition.h"
#include "ae108/cpppetsc/Mesh.h"
#include "ae108/cpppetsc/SequentialComputePolicy.h"
#include "ae108/elements/mesh/read_mesh_from_file.h"
#include "ae108/elements/tensor/as_matrix_of_rows.h"
#include "ae108/elements/tensor/as_vector.h"
#include <Eigen/Dense>

#include "ae108/cpppetsc/Viewer.h"
#include "ae108/cpppetsc/createVectorFromSource.h"
#include "ae108/cpppetsc/setName.h"
#include "ae108/cpppetsc/writeToViewer.h"

#include "ae108/elements/TimoshenkoBeamElement.h"
#include "ae108/solve/boundaryConditionsToTransform.h"

#include "ae108/homogenization/RVEBase.h"
#include "ae108/homogenization/assembly/plugins/AssembleEffectiveTangentMatrixPlugin.h"
#include "ae108/homogenization/assembly/plugins/AssembleTractionAverageStressPlugin.h"
#include "ae108/homogenization/materialmodels/Homogenization.h"
#include "ae108/homogenization/materialmodels/compute_displacements.h"
#include "ae108/homogenization/materialmodels/compute_effective_tangent_matrix.h"
#include "ae108/homogenization/materialmodels/compute_traction_average_stress.h"
#include "ae108/homogenization/utilities/assign_lattice_to.h"
#include "ae108/solve/boundaryConditionsToTransform.h"

#include "utilities/read_properties_from_file.h"

#include <iostream>
#include <algorithm>
#include <chrono>

const static Eigen::IOFormat TangentMatrixFormat(Eigen::StreamPrecision,
                                                 Eigen::DontAlignCols, ", ",
                                                 ", ", "", "", "", "\n");

using Policy = ae108::cpppetsc::SequentialComputePolicy;
using Mesh = ae108::cpppetsc::Mesh<Policy>;
using Viewer = ae108::cpppetsc::Viewer<Policy>;

constexpr auto dimension = Mesh::size_type{2};
constexpr auto number_of_vertices_per_element = Mesh::size_type{2};

using Connectivity =
    std::vector<std::array<std::size_t, number_of_vertices_per_element>>;

using Point = std::array<Mesh::value_type, dimension>;

using VertexPositions = std::vector<Point>;

using LatticeVectors =
    std::array<std::array<Mesh::value_type, dimension>, dimension>;

using Element = ae108::elements::TimoshenkoBeamElement<dimension>;

using Properties =
    ae108::elements::TimoshenkoBeamProperties<Mesh::value_type, dimension>;

using RVE = ae108::homogenization::RVEBase<Policy, Element>;

constexpr auto density = Mesh::value_type(1.);

int main(int argc, char *argv[])
{
  auto tbegin = std::chrono::steady_clock::now(); // start timer
  bool verbose = false;
  std::string N;
  std::string name;
  Policy::handleError(PetscInitialize(&argc, &argv, NULL, NULL));
  {
    ae108::cmdline::CommandLineOptionParser(std::cerr)
        .withOption("N", "Unit cell size.", &N)
        .withOption("name,n", "Unit cell name.", &name)
        .parse(argc, argv);
  }

  size_t nRealizations = 100;

  // read mesh
  auto geo = ae108::elements::mesh::read_mesh_from_file<Point>(
      "mesh");

  // read properties
  // auto parameters = read_parameters_from_file(
    //  "../../../../disordered-trusses/data/N=" + N + "/" + name + "_r=" + std::to_string(r) + "-prop.dat");

  // auto properties = Properties{parameters[0], parameters[1],
      //                          parameters[2], parameters[3], parameters[4]};
  auto properties = Properties{200, 80,
                                0.8, 1, 1};
  // find lattice vectors
  auto corners = std::minmax_element(geo.positions().begin(), geo.positions().end(),
                                      [](const Point &lhs, const Point &rhs)
                                      {
                                        return ae108::elements::tensor::as_vector(&lhs).norm() < ae108::elements::tensor::as_vector(&rhs).norm();
                                      });
  auto lattice_vectors =
      LatticeVectors{{{corners.second[0][0] - corners.first[0][0], 0.}, {0., corners.second[0][1] - corners.first[0][1]}}};

  auto mesh = std::unique_ptr<typename RVE::mesh_type>(
      new typename RVE::mesh_type(Mesh::fromConnectivity(
          dimension, geo.connectivity(), geo.positions().size(),
          Element::degrees_of_freedom())));

  auto element_source = ae108::homogenization::utilities::assign_lattice_to(
      geo.connectivity(), geo.positions(), lattice_vectors);

  double mass = 0;
  auto assembler = RVE::assembler_type();
  for (const auto &element : mesh->localElements())
  {

    std::array<typename Mesh::value_type, dimension> element_axis;
    ae108::elements::tensor::as_vector(&element_axis) =
        ae108::elements::tensor::as_vector(&geo.positions().at(
            geo.connectivity().at(element.index()).at(1))) -
        ae108::elements::tensor::as_vector(&geo.positions().at(
            geo.connectivity().at(element.index()).at(0)));

    const auto weight =
        1. / std::count(element_source.begin(), element_source.end(),
                        element_source[element.index()]);
    mass += ae108::elements::tensor::as_vector(&element_axis).norm() *
            density * weight;

    assembler.emplaceElement(
        element,
        timoshenko_beam_stiffness_matrix(element_axis, properties) * weight);
  }

  const auto volume =
      ae108::elements::tensor::as_matrix_of_rows(&lattice_vectors)
          .determinant();

  const auto vertex_source =
      ae108::homogenization::utilities::assign_lattice_to(geo.positions(),
                                                          lattice_vectors);

  RVE rve{geo.positions(), std::move(mesh), assembler, vertex_source, volume};

  auto periodic_displacement_bc =
      [&](const typename ae108::homogenization::materialmodels::
              Homogenization<RVE>::rve_type &rve,
          const typename ae108::homogenization::materialmodels::
              Homogenization<RVE>::DisplacementGradient
                  &displacement_gradient)
  {
    using mesh_type =
        typename ae108::homogenization::materialmodels::Homogenization<
            RVE>::rve_type::mesh_type;
    using bc_type =
        ae108::cpppetsc::GeneralizedMeshBoundaryCondition<mesh_type>;
    using size_type = typename bc_type::size_type;

    std::vector<bc_type> boundary_conditions;
    for (const auto &target : rve.mesh->localVertices())
    {
      const auto &source = rve.source_map[target.index()];
      if (source > 0 && source != target.index())
      { // PBC
        const auto displacement =
            tensor::as_matrix_of_rows(&displacement_gradient) *
            (tensor::as_vector(&rve.vertex_positions[target.index()]) -
              tensor::as_vector(&rve.vertex_positions[source]));
        for (size_type dof = 0; dof < target.numberOfDofs(); dof++)
        {
          boundary_conditions.push_back(
              {{target.index(), dof},
                {
                    {1., {size_type(source), dof}},
                },
                displacement[dof]});
        }
      }
      else if (!source) // Dirichlet Boundary Condition
      {
        const auto displacement =
            tensor::as_matrix_of_rows(&displacement_gradient) *
            tensor::as_vector(&rve.vertex_positions[target.index()]);
        for (size_type dof = 0; dof < 2; dof++) // enforce affine displacements
          boundary_conditions.push_back(
              {{target.index(), dof}, {}, displacement[dof]});
        if (rve.source_map[target.index()] != target.index())
          for (size_type dof = 2; dof < target.numberOfDofs();
                dof++) // make rotational dofs periodic
            boundary_conditions.push_back(
                {{target.index(), dof},
                  {
                      {1.,
                      {size_type(rve.source_map[target.index()]), dof}},
                  },
                  displacement[dof]});
      }
    }
    return boundary_conditions;
  };

  const ae108::homogenization::materialmodels::Homogenization<RVE> model(
      std::move(rve), periodic_displacement_bc);

  const auto displacement_gradient =
      ae108::homogenization::materialmodels::Homogenization<
          RVE>::DisplacementGradient{{{0., 0.}, {0., 0.}, {0., 0.}}};

  const auto tangent_matrix =
      ae108::homogenization::materialmodels::compute_effective_tangent_matrix(
          model, 0, displacement_gradient, 0.);

  const auto tangentMatrix =
      ae108::elements::tensor::as_two_tensor(&tangent_matrix).eval();

  Eigen::Matrix<double, 3, 3> voigtMatrix =
      Eigen::Matrix<double, 3, 3>::Zero();
  voigtMatrix(0, 0) = tangentMatrix(0, 0);                                                                           //C_1111
  voigtMatrix(1, 1) = tangentMatrix(3, 3);                                                                           //C_2222
  voigtMatrix(2, 2) = (tangentMatrix(1, 1) + tangentMatrix(2, 2) + tangentMatrix(1, 2) + tangentMatrix(2, 1)) / 4.0; //C_1212
  voigtMatrix(0, 1) = tangentMatrix(0, 3);                                                                           //C_1122
  voigtMatrix(1, 0) = tangentMatrix(3, 0);                                                                           //C_2211
  voigtMatrix(0, 2) = (tangentMatrix(0, 1) + tangentMatrix(0, 2)) / 2.0;                                             //C_1112
  voigtMatrix(2, 0) = (tangentMatrix(1, 0) + tangentMatrix(2, 0)) / 2.0;                                             //C_1211
  voigtMatrix(1, 2) = (tangentMatrix(3, 1) + tangentMatrix(3, 2)) / 2.0;                                             //C_2212
  voigtMatrix(2, 1) = (tangentMatrix(1, 3) + tangentMatrix(2, 3)) / 2.0;                                             //C_1222

  //if (Policy::isPrimaryRank())
  if (true)
  {
    if (verbose == true)
    {
      std::cout << "Voigt tangent matrix" << std::endl;
      std::cout << voigtMatrix << std::endl;
    }
    std::string fname =
        "tangent_matrix";
    std::ofstream file;
    file.open(fname, std::ofstream::app);

    
    file << "volume, "
          << "density";
    for (size_t rowID = 1; rowID <= 3; rowID++)
      for (size_t colID = 1; colID <= 3; colID++)
        file << ", C" << rowID << colID;
    file << std::endl;
    

    file << volume << ", " << mass / volume << ", ";
    file << voigtMatrix.format(TangentMatrixFormat);
    file.close();
  }

  bool saveMesh = false;
  if (saveMesh == true)
  {
    // First we collect the coordinates in a vector.
    using DataSource = std::function<void(Mesh::size_type, Mesh::value_type *)>;
    auto coordinates = ae108::cpppetsc::createVectorFromSource(
        *model.rve().mesh, dimension,
        DataSource(
            [&](const Mesh::size_type index, Mesh::value_type *const out)
            {
              const auto &position = model.rve().vertex_positions.at(index);
              std::copy(position.begin(), position.end(), out);
            }));
    ae108::cpppetsc::setName("coordinates", &coordinates);

    // Now we write the mesh to a file.
    auto viewer = Viewer::fromHdf5FilePath(
        (name + std::string(".ae108")).c_str(), Viewer::Mode::write);
    ae108::cpppetsc::writeToViewer(*model.rve().mesh, coordinates, &viewer);

    if (Policy::isPrimaryRank())
      fprintf(stderr,
              "The mesh has been written to the file "
              "\"%s.ae108\".\n",
              name.c_str());
  }
  fprintf(stderr, "Done with "
                  "\"%s\".\n",
          name.c_str());
 // std::cout << "Total time = ";
  //           << std::chrono::duration_cast<std::chrono::seconds>(
  //                  std::chrono::steady_clock::now() - tbegin)
  //                  .count()
  //           << "[s]" << std::endl;

  return 0;
}