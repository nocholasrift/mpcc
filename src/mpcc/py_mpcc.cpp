#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <mpcc/map_util.h>
#include <mpcc/mpcc_core.h>
#include <mpcc/utils.h>

#include <mpcc/tube_gen.h>
#include "mpcc/types.h"

using namespace mpcc;

using Polynomial   = types::Polynomial;
using Spline       = types::Spline;
using StateHorizon = types::StateHorizon;
using InputHorizon = types::InputHorizon;
// using MPCHorizon   = types::MPCHorizon;
using Trajectory = types::Trajectory;
using View       = Trajectory::View;

namespace py = pybind11;
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::VectorXd>);

PYBIND11_MODULE(py_mpcc, m) {
  py::bind_vector<std::vector<Eigen::VectorXd>>(m, "vec_VecXd");

  py::enum_<MPCType>(m, "MPCType")
      .value("DOUBLE_INTEGRATOR", MPCType::kDoubleIntegrator)
      .value("UNICYCLE", MPCType::kUnicycle);

  py::class_<Polynomial>(m, "Polynomial")
      .def(py::init<>())
      .def(py::init<const Eigen::VectorXd&>())
      .def("set_coeffs", static_cast<void (Polynomial::*)(Eigen::VectorXd&)>(
                             &Polynomial::set_coeffs))
      .def("get_coeffs", &Polynomial::get_coeffs)
      .def("derivative", &Polynomial::derivative)
      .def("pos", &Polynomial::pos);

  py::class_<Spline>(m, "Spline")
      .def(py::init<>())
      .def(py::init<const Eigen::RowVectorXd&, const Eigen::RowVectorXd&>())
      .def("pos", &Spline::pos)
      .def("derivative", &Spline::derivative)
      .def("get_knots", &Spline::get_knots)
      .def("get_ctrls", &Spline::get_ctrls);

  py::class_<View>(m, "View")
      .def_readwrite("knots", &View::knots)
      .def_readwrite("xs", &View::xs)
      .def_readwrite("ys", &View::ys)
      .def_readwrite("arclen", &View::arclen);

  py::class_<Trajectory>(m, "Trajectory")
      .def(py::init<>())
      .def(py::init<const Spline&, const Spline&>())
      .def("get_arclen", &Trajectory::get_arclen)
      .def("get_adjusted_traj", &Trajectory::get_adjusted_traj)
      .def("get_closest_s", &Trajectory::get_closest_s)
      .def("get_ctrls_x", &Trajectory::get_ctrls_x)
      .def("get_ctrls_y", &Trajectory::get_ctrls_y)
      .def("view", &Trajectory::view)
      .def("__call__",
           static_cast<Trajectory::Point (Trajectory::*)(double) const>(
               &Trajectory::operator()))
      .def("__call__",
           static_cast<Trajectory::Point (Trajectory::*)(double, unsigned int)
                           const>(&Trajectory::operator()));

  py::class_<StateHorizon>(m, "StateHorizon")
      .def_readwrite("xs", &StateHorizon::xs)
      .def_readwrite("ys", &StateHorizon::ys)
      .def_readwrite("arclens", &StateHorizon::arclens)
      .def_readwrite("arclens_dot", &StateHorizon::arclens_dot);

  py::class_<InputHorizon>(m, "InputHorizon")
      .def_readwrite("arclens_ddot", &InputHorizon::arclens_ddot);

  // Unicycle specific horizons
  py::class_<UnicycleMPCC::StateHorizon, StateHorizon>(m,
                                                       "UnicycleStateHorizon")
      .def_readwrite("thetas", &UnicycleMPCC::StateHorizon::thetas)
      .def_readwrite("vs", &UnicycleMPCC::StateHorizon::vs);

  py::class_<UnicycleMPCC::InputHorizon, InputHorizon>(m,
                                                       "UnicycleInputHorizon")
      .def_readwrite("angvels", &UnicycleMPCC::InputHorizon::angvels)
      .def_readwrite("linaccs", &UnicycleMPCC::InputHorizon::linaccs);

  py::class_<UnicycleMPCC::MPCHorizon>(m, "UnicycleHorizon")
      .def_readwrite("states", &UnicycleMPCC::MPCHorizon::states)
      .def_readwrite("inputs", &UnicycleMPCC::MPCHorizon::inputs)
      .def_readwrite("length", &UnicycleMPCC::MPCHorizon::length)
      .def("get_state_at_step", &UnicycleMPCC::MPCHorizon::get_state_at_step)
      .def("get_input_at_step", &UnicycleMPCC::MPCHorizon::get_input_at_step);

  // Double Integrator specific horizons
  py::class_<DIMPCC::StateHorizon, StateHorizon>(m, "DIStateHorizon")
      .def_readwrite("vs_x", &DIMPCC::StateHorizon::vs_x)
      .def_readwrite("vs_y", &DIMPCC::StateHorizon::vs_y);

  py::class_<DIMPCC::InputHorizon, InputHorizon>(m, "DIInputHorizon")
      .def_readwrite("accs_x", &DIMPCC::InputHorizon::accs_x)
      .def_readwrite("accs_y", &DIMPCC::InputHorizon::accs_y);

  py::class_<DIMPCC::MPCHorizon>(m, "DIHorizon")
      .def_readwrite("states", &DIMPCC::MPCHorizon::states)
      .def_readwrite("inputs", &DIMPCC::MPCHorizon::inputs)
      .def_readwrite("length", &DIMPCC::MPCHorizon::length)
      .def("get_state_at_step", &DIMPCC::MPCHorizon::get_state_at_step)
      .def("get_input_at_step", &DIMPCC::MPCHorizon::get_input_at_step);

  py::class_<MPCCore>(m, "MPCCore")
      .def(py::init<>())
      .def(py::init<const MPCType&>())
      .def("load_params", &MPCCore::load_params)
      .def("get_params", &MPCCore::get_params)
      .def("set_map", &MPCCore::set_map<unsigned char>)
      .def("set_odom", &MPCCore::set_odom)
      .def("set_trajectory",
           (void (MPCCore::*)(const Eigen::VectorXd&, const Eigen::VectorXd&,
                              const Eigen::VectorXd&))&MPCCore::set_trajectory)
      .def("get_tube", &MPCCore::get_tube)
      .def("get_cbf_data", &MPCCore::get_cbf_data)
      .def("get_horizon", &MPCCore::get_horizon)
      .def("solve", &MPCCore::solve)
      .def("get_trajectory", &MPCCore::get_trajectory)
      .def("get_non_extended_trajectory", &MPCCore::get_non_extended_trajectory)
      .def("get_solver_status", &MPCCore::get_solver_status)
      .def("get_cbf_data", &MPCCore::get_cbf_data)
      .def("get_input_limits", &MPCCore::get_input_limits)
      .def("get_state_limits", &MPCCore::get_state_limits)
      .def("get_state", &MPCCore::get_state);

  py::class_<map_util::IGrid, std::shared_ptr<map_util::IGrid>>(m, "IGrid")
      .def("world_to_map", &map_util::IGrid::world_to_map)
      .def("map_to_world", &map_util::IGrid::map_to_world)
      .def("index_to_cells", &map_util::IGrid::index_to_cells)
      .def("cells_to_index", &map_util::IGrid::cells_to_index)
      .def("get_origin", &map_util::IGrid::get_origin)
      .def("get_resolution", &map_util::IGrid::get_resolution)
      .def("get_size", &map_util::IGrid::get_size)
      .def("is_occupied", py::overload_cast<unsigned int, const std::string&>(
                              &map_util::IGrid::is_occupied, py::const_))
      .def("clamp_point_to_bounds", &map_util::IGrid::clamp_point_to_bounds)
      .def("get_occupied", &map_util::IGrid::get_occupied);

  // python will only have access to the unsigned char version for now :)
  using OccupancyGrid = map_util::OccupancyGrid<unsigned char>;
  py::class_<OccupancyGrid::MapConfig>(m, "MapConfig")
      .def(py::init<>())
      .def_readwrite("width", &OccupancyGrid::MapConfig::width)
      .def_readwrite("height", &OccupancyGrid::MapConfig::height)

      .def_readwrite("resolution", &OccupancyGrid::MapConfig::resolution)
      .def_readwrite("origin", &OccupancyGrid::MapConfig::origin)

      .def_readwrite("occupied_values",
                     &OccupancyGrid::MapConfig::occupied_values)
      .def_readwrite("no_information_values",
                     &OccupancyGrid::MapConfig::no_information_values);

  py::class_<OccupancyGrid, map_util::IGrid, std::shared_ptr<OccupancyGrid>>(
      m, "OccupancyGrid")
      .def(py::init<>())
      .def(py::init<const OccupancyGrid::MapConfig&,
                    const std::vector<unsigned char>&>())
      .def("get_cost", py::overload_cast<unsigned int, const std::string&>(
                           &OccupancyGrid::get_cost, py::const_))
      .def("update", py::overload_cast<int, int, double, double, double,
                                       const std::vector<unsigned char>&,
                                       const std::vector<unsigned char>&,
                                       const std::vector<unsigned char>&>(
                         &OccupancyGrid::update))
      .def("get_occupied_values", &OccupancyGrid::get_occupied_values);

  // utilities
  m.def("extend_trajectory", &utils::extend_trajectory);
}
