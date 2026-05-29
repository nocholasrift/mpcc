#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <mpcc/common/map_util.h>
#include <mpcc/common/mpcc_base.h>
#include <mpcc/common/mpcc_config.h>
#include <mpcc/common/mpcc_core.h>
#include <mpcc/common/tube_gen.h>
#include <mpcc/common/utils.h>
#include "mpcc/common/types.h"

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
      .value("UNICYCLE", MPCType::kUnicycle)
      .value("BICYCLE", MPCType::kBicycle);

  py::class_<Polynomial>(m, "Polynomial")
      .def(py::init<>())
      .def(py::init<const Eigen::VectorXd&>())
      .def("__call__",
           py::overload_cast<double>(&Polynomial::operator(), py::const_))
      .def("__call__", py::overload_cast<double, unsigned int>(
                           &Polynomial::operator(), py::const_))
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

  // Bicycle specific horizons
  py::class_<BicycleMPCC::StateHorizon, StateHorizon>(m, "BicycleStateHorizon")
      .def_readwrite("thetas", &BicycleMPCC::StateHorizon::thetas)
      .def_readwrite("vs", &BicycleMPCC::StateHorizon::vs);

  py::class_<BicycleMPCC::InputHorizon, InputHorizon>(m, "BicycleInputHorizon")
      .def_readwrite("accs", &BicycleMPCC::InputHorizon::accs)
      .def_readwrite("deltas", &BicycleMPCC::InputHorizon::deltas);

  py::class_<BicycleMPCC::MPCHorizon>(m, "BicycleHorizon")
      .def_readwrite("states", &BicycleMPCC::MPCHorizon::states)
      .def_readwrite("inputs", &BicycleMPCC::MPCHorizon::inputs)
      .def_readwrite("length", &BicycleMPCC::MPCHorizon::length)
      .def("get_state_at_step", &BicycleMPCC::MPCHorizon::get_state_at_step)
      .def("get_input_at_step", &BicycleMPCC::MPCHorizon::get_input_at_step);

  py::enum_<SolverStatus>(m, "SolverStatus")
      .value("Success", SolverStatus::kSuccess)
      .value("Presolve", SolverStatus::kPresolve)
      .value("ParamMismatch", SolverStatus::kParamMismatch)
      .value("SolverNotReady", SolverStatus::kSolverNotReady);

  py::class_<MPCResult>(m, "MPCResult")
      .def_readwrite("status", &MPCResult::status)
      .def_readwrite("command", &MPCResult::command);

  // MPC Config
  py::class_<mpcc::CostWeights, std::shared_ptr<mpcc::CostWeights>>(
      m, "CostWeights")
      .def(py::init<>())
      .def_readwrite("w_vel", &mpcc::CostWeights::w_vel)
      .def_readwrite("w_angvel", &mpcc::CostWeights::w_angvel)
      .def_readwrite("w_linvel", &mpcc::CostWeights::w_linvel)
      .def_readwrite("w_angvel_d", &mpcc::CostWeights::w_angvel_d)
      .def_readwrite("w_linvel_d", &mpcc::CostWeights::w_linvel_d)
      .def_readwrite("w_etheta", &mpcc::CostWeights::w_etheta)
      .def_readwrite("w_cte", &mpcc::CostWeights::w_cte)
      .def_readwrite("w_lag_e", &mpcc::CostWeights::w_lag_e)
      .def_readwrite("w_contour_e", &mpcc::CostWeights::w_contour_e)
      .def_readwrite("w_speed", &mpcc::CostWeights::w_speed)
      .def("__repr__", [](const mpcc::CostWeights& w) {
        return "<CostWeights w_lag_e=" + std::to_string(w.w_lag_e) +
               " w_contour_e=" + std::to_string(w.w_contour_e) +
               " w_speed=" + std::to_string(w.w_speed) + ">";
      });

  py::class_<mpcc::Constraints, std::shared_ptr<mpcc::Constraints>>(
      m, "Constraints")
      .def(py::init<>())
      .def_readwrite("max_angvel", &mpcc::Constraints::max_angvel)
      .def_readwrite("max_linvel", &mpcc::Constraints::max_linvel)
      .def_readwrite("max_linacc", &mpcc::Constraints::max_linacc)
      .def_readwrite("max_angacc", &mpcc::Constraints::max_angacc)
      .def_readwrite("bound_value", &mpcc::Constraints::bound_value)
      .def("__repr__", [](const mpcc::Constraints& c) {
        return "<Constraints max_linvel=" + std::to_string(c.max_linvel) +
               " max_angvel=" + std::to_string(c.max_angvel) + ">";
      });

  py::class_<mpcc::CBFConfig, std::shared_ptr<mpcc::CBFConfig>>(m, "CBFConfig")
      .def(py::init<>())
      .def_readwrite("use_cbf", &mpcc::CBFConfig::use_cbf)
      .def_readwrite("alpha_abv", &mpcc::CBFConfig::alpha_abv)
      .def_readwrite("alpha_blw", &mpcc::CBFConfig::alpha_blw)
      .def_readwrite("colinear", &mpcc::CBFConfig::colinear)
      .def_readwrite("padding", &mpcc::CBFConfig::padding)
      .def_readwrite("dynamic_alpha", &mpcc::CBFConfig::dynamic_alpha)
      .def_readwrite("min_alpha", &mpcc::CBFConfig::min_alpha)
      .def_readwrite("max_alpha", &mpcc::CBFConfig::max_alpha)
      .def_readwrite("min_alpha_dot", &mpcc::CBFConfig::min_alpha_dot)
      .def_readwrite("max_alpha_dot", &mpcc::CBFConfig::max_alpha_dot)
      .def_readwrite("min_h_val", &mpcc::CBFConfig::min_h_val)
      .def_readwrite("max_h_val", &mpcc::CBFConfig::max_h_val)
      .def("__repr__", [](const mpcc::CBFConfig& c) {
        return "<CBFConfig use_cbf=" +
               std::string(c.use_cbf ? "True" : "False") +
               " alpha_abv=" + std::to_string(c.alpha_abv) +
               " alpha_blw=" + std::to_string(c.alpha_blw) + ">";
      });

  py::class_<mpcc::CLFConfig, std::shared_ptr<mpcc::CLFConfig>>(m, "CLFConfig")
      .def(py::init<>())
      .def_readwrite("w_lag_e", &mpcc::CLFConfig::w_lag_e)
      .def_readwrite("w_contour_e", &mpcc::CLFConfig::w_contour_e)
      .def_readwrite("gamma", &mpcc::CLFConfig::gamma)
      .def("__repr__", [](const mpcc::CLFConfig& c) {
        return "<CLFConfig gamma=" + std::to_string(c.gamma) +
               " w_lag_e=" + std::to_string(c.w_lag_e) + ">";
      });

  py::class_<mpcc::TubeConfig, std::shared_ptr<mpcc::TubeConfig>>(m,
                                                                  "TubeConfig")
      .def(py::init<>())
      .def_readwrite("poly_degree", &mpcc::TubeConfig::poly_degree)
      .def_readwrite("num_samples", &mpcc::TubeConfig::num_samples)
      .def_readwrite("max_width", &mpcc::TubeConfig::max_width)
      .def("__repr__", [](const mpcc::TubeConfig& t) {
        return "<TubeConfig poly_degree=" + std::to_string(t.poly_degree) +
               " max_width=" + std::to_string(t.max_width) + ">";
      });

  py::class_<mpcc::PropControllerConfig,
             std::shared_ptr<mpcc::PropControllerConfig>>(
      m, "PropControllerConfig")
      .def(py::init<>())
      .def_readwrite("gain", &mpcc::PropControllerConfig::gain)
      .def_readwrite("gain_thresh", &mpcc::PropControllerConfig::gain_thresh)
      .def("__repr__", [](const mpcc::PropControllerConfig& p) {
        return "<PropControllerConfig gain=" + std::to_string(p.gain) +
               " gain_thresh=" + std::to_string(p.gain_thresh) + ">";
      });

  py::class_<mpcc::MPCConfig, std::shared_ptr<mpcc::MPCConfig>>(m, "MPCConfig")
      .def(py::init<>())
      .def("copy",
           [](const mpcc::MPCConfig& self) {
             return std::make_shared<mpcc::MPCConfig>(self);
           })
      .def_readwrite("steps", &mpcc::MPCConfig::steps)
      .def_readwrite("dt", &mpcc::MPCConfig::dt)
      .def_readwrite("ref_samples", &mpcc::MPCConfig::ref_samples)
      .def_readwrite("ref_length", &mpcc::MPCConfig::ref_length)
      .def_readwrite("input_type", &mpcc::MPCConfig::input_type)
      .def_readwrite("body_length", &mpcc::MPCConfig::body_length)
      .def_readwrite("weights", &mpcc::MPCConfig::weights)
      .def_readwrite("constraints", &mpcc::MPCConfig::constraints)
      .def_readwrite("cbf", &mpcc::MPCConfig::cbf)
      .def_readwrite("clf", &mpcc::MPCConfig::clf)
      .def_readwrite("tube", &mpcc::MPCConfig::tube)
      .def_readwrite("prop", &mpcc::MPCConfig::prop)
      .def("__repr__", [](const mpcc::MPCConfig& c) {
        return "<MPCConfig steps=" + std::to_string(c.steps) +
               " dt=" + std::to_string(c.dt) +
               " ref_samples=" + std::to_string(c.ref_samples) + ">";
      });

  py::class_<MPCCore>(m, "MPCCore")
      // .def(py::init<>())
      .def(py::init<std::shared_ptr<mpcc::MPCConfig>>())
      .def("load_params", &MPCCore::load_params)
      .def("get_params", &MPCCore::get_params)
      .def("set_map", &MPCCore::set_map<unsigned char>)
      .def("set_odom", &MPCCore::set_odom)
      .def("set_trajectory",
           (void (MPCCore::*)(const Eigen::VectorXd&, const Eigen::VectorXd&,
                              const Eigen::VectorXd&))&MPCCore::set_trajectory)
      .def("get_tube", &MPCCore::get_tube)
      .def("get_horizon", &MPCCore::get_horizon)
      .def("solve", &MPCCore::solve)
      .def("get_trajectory", &MPCCore::get_trajectory)
      .def("get_non_extended_trajectory", &MPCCore::get_non_extended_trajectory)
      .def("get_solver_status", &MPCCore::get_solver_status)
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
