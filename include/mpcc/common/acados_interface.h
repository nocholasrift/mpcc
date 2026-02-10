#ifndef MPCC_ACADOS_INTERFACE_H
#define MPCC_ACADOS_INTERFACE_H

#include <iostream>
#include <memory>
#include <vector>

// acados
#include <acados_c/ocp_nlp_interface.h>

#include <Eigen/Core>

namespace mpcc {

// RAII interface for ACADOS solver
template <typename MPCTraits>
class AcadosInterface {

 public:
  AcadosInterface() {}
  ~AcadosInterface() noexcept {}

  int initialize(double mpc_steps);
  int solve();

  void update_params(unsigned int stage, const std::vector<double>& params);

  void set_model_constraint(unsigned int stage, const std::string& field,
                            const double* data);

  void set_output(unsigned int stage, const std::string& field,
                  const double* data);

  void get_output(unsigned int stage, const std::string& field,
                  double* data) const;

  double get_mpc_steps() const { return mpc_steps_; }

  double is_initialized() const { return is_initialized_; }

 private:
  AcadosInterface(AcadosInterface const&);
  AcadosInterface& operator=(AcadosInterface const&);

  void get_ocp_views();

  // passing member function to unique_ptr was not pretty, using these structs
  // instead.
  struct CapsuleDeleter {
    void operator()(typename MPCTraits::SolverCapsule* p) const noexcept {
      if (p)
        MPCTraits::free_capsule(p);
    }
  };

  struct CMallocDeleter {
    void operator()(double* p) const noexcept { std::free(p); }
  };

 private:
  std::unique_ptr<typename MPCTraits::SolverCapsule, CapsuleDeleter>
      acados_ocp_capsule_;

  // views that are all managed by the acados capsule.
  ocp_nlp_in* nlp_in_{nullptr};
  ocp_nlp_out* nlp_out_{nullptr};
  ocp_nlp_dims* nlp_dims_{nullptr};
  ocp_nlp_config* nlp_config_{nullptr};
  ocp_nlp_solver* nlp_solver_{nullptr};
  void* nlp_opts_{nullptr};

  // apparently you must call FREE on memory allocated with MALLOC...
  // I was not aware of this :)
  std::unique_ptr<double, CMallocDeleter> new_time_steps_;

  double mpc_steps_{0.};
  bool is_initialized_{false};
};

// implementation

template <typename MPCTraits>
int AcadosInterface<MPCTraits>::initialize(double mpc_steps) {
  using Capsule_t = typename MPCTraits::SolverCapsule;

  // if mpc_steps is the same as what was set before, there is
  // no need to reinitialize the capsule. Just "pretend" we initialized
  // correctly and keep using old one.
  if (mpc_steps_ == mpc_steps) {
    return 0;
  }

  double* time_steps = nullptr;
  Capsule_t* capsule;

  // this function can cause termination if internal "precomputations"
  // fail...
  int status = MPCTraits::create_capsule(mpc_steps, time_steps, capsule);

  if (status) {
    is_initialized_ = false;
    return status;
  }

  // reset will delete old capsule automatically
  acados_ocp_capsule_.reset(capsule);
  new_time_steps_.reset(time_steps);

  get_ocp_views();

  mpc_steps_      = mpc_steps;
  is_initialized_ = true;

  return status;
}

template <typename MPCTraits>
void AcadosInterface<MPCTraits>::get_ocp_views() {
  auto* capsule_ptr = acados_ocp_capsule_.get();
  nlp_in_           = MPCTraits::get_nlp_in(capsule_ptr);
  nlp_out_          = MPCTraits::get_nlp_out(capsule_ptr);
  nlp_opts_         = MPCTraits::get_nlp_opts(capsule_ptr);
  nlp_dims_         = MPCTraits::get_nlp_dims(capsule_ptr);
  nlp_solver_       = MPCTraits::get_nlp_solver(capsule_ptr);
  nlp_config_       = MPCTraits::get_nlp_config(capsule_ptr);
}

template <typename MPCTraits>
int AcadosInterface<MPCTraits>::solve() {
  return MPCTraits::solve(acados_ocp_capsule_.get());
}

template <typename MPCTraits>
void AcadosInterface<MPCTraits>::update_params(
    unsigned int stage, const std::vector<double>& params) {
  MPCTraits::set_params(acados_ocp_capsule_.get(), stage, params);
}

template <typename MPCTraits>
void AcadosInterface<MPCTraits>::set_model_constraint(unsigned int stage,
                                                      const std::string& field,
                                                      const double* data) {

  ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, stage,
                                field.c_str(), const_cast<double*>(data));
}

template <typename MPCTraits>
void AcadosInterface<MPCTraits>::set_output(unsigned int stage,
                                            const std::string& field,
                                            const double* data) {
  ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, stage, field.c_str(),
                  const_cast<double*>(data));
}

template <typename MPCTraits>
void AcadosInterface<MPCTraits>::get_output(unsigned int stage,
                                            const std::string& field,
                                            double* data) const {
  ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, stage, field.c_str(), data);
}

}  // namespace mpcc
#endif
