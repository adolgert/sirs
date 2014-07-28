#ifndef _SEASONAL_INTERVAL_H_
#define _SEASONAL_INTERVAL_H_ 1

#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>

#include "gsl/gsl_errno.h"
#include "gsl/gsl_math.h"
#include "gsl/gsl_roots.h"
#include "gspn_random.hpp"
#include "smv.hpp"


double SeasonalInterval(double t0, double ti, double b0, double b1, double U);
double TestSeasonal(double b1);


namespace seasonal {
struct SeasonalParams {
  double X; // -ln(U)+t0+(b1/2pi)*sin(2pi(t0-ti))
  double beta; // b1/2pi
  double ti; // Offset for seasonality.
};

double seasonal_hazard(double t, void* params);
double seasonal_hazard_df(double t, void* params);
void seasonal_hazard_fdf(double t, void* params,
    double* f, double* df);
}


template<typename RNG>
class SeasonalBeta : public afidd::smv::TransitionDistribution<RNG>
{
  double b0_; // beta0
  double b1_; // beta1
  double ti_; // Offset of seasonality into year.
  double te_; // enabling_time
  gsl_function* seasonal_function_;
  seasonal::SeasonalParams* params_;
  const gsl_root_fsolver_type* solver_type_;
  gsl_root_fsolver* solver_;

public:
  SeasonalBeta(double b0, double b1, double ti, double te)
      : b0_(b0), b1_(b1), ti_(ti), te_(te),
      solver_type_(gsl_root_fsolver_brent),
      params_(new seasonal::SeasonalParams()),
      seasonal_function_(new gsl_function())
  {
    params_->beta=b1_/(2*M_PI);
    params_->ti=ti_;

    solver_=gsl_root_fsolver_alloc(solver_type_);
    seasonal_function_->function=&seasonal::seasonal_hazard;
    seasonal_function_->params=params_;
  }
  ~SeasonalBeta() {
    gsl_root_fsolver_free(solver_);
    delete params_;
    delete seasonal_function_;
  }

  virtual double Sample(double current_time, RNG& rng) const {
    return ImplicitHazardIntegral(-std::log(afidd::smv::uniform(rng)),
      current_time);
  }

  virtual double EnablingTime() const { return te_; }

  // Whether the hazard is always bounded below infinity.
  virtual bool BoundedHazard() const { return true; }

  // Integral of hazard from absolute time t0 to t1.
  virtual double HazardIntegral(double t0, double t1) const {
    return b0_*((t1-t0)+(b1_/(2*M_PI))*
              (std::sin(2*M_PI*(t1-ti_)) - std::sin(2*M_PI*(t0-ti_))));
  }

  // xa = int_t0^t hazard(s, te) ds.  Solve for t.
  virtual double ImplicitHazardIntegral(double xa, double t0) const {
    SMVLOG(BOOST_LOG_TRIVIAL(trace) << "Seasonal implicit t0 " << t0);
    double resolution=1e-9;
    double bounds_err=1e-13;
    int iter_max=1000;
    double t=0;
    params_->X=xa/b0_ + t0 + (b1_/(2*M_PI))*std::sin(2*M_PI*(t0-ti_));

    double low=params_->X-b1_/(2*M_PI)-bounds_err;
    double high=params_->X+b1_/(2*M_PI)+bounds_err;
    //std::cout << "low2 " << low << " high " << high << std::endl;
    //std::cout << "X2 " << params_->X << " b " << params_->beta << " ti "
    //  << params_->ti << std::endl;
    //std::cout << "bracklow2 " << (seasonal_function_->function)(
    //  low, seasonal_function_->params) << std::endl;
    //std::cout << "brackhigh2 " << (seasonal_function_->function)(
    //  high, seasonal_function_->params) << std::endl;

    gsl_root_fsolver_set(solver_, seasonal_function_, low, high);

    int test_status=GSL_CONTINUE;
    int iter=0;
    while (test_status==GSL_CONTINUE && iter<iter_max) {
      int status=gsl_root_fsolver_iterate(solver_);
      switch (status) {
        case 0:
          break;
        case GSL_EBADFUNC:
          BOOST_LOG_TRIVIAL(error) << "Bad function";
          break;
        case GSL_EZERODIV:
          BOOST_LOG_TRIVIAL(error) << "Divide by zero";
          break;
        default:
          BOOST_LOG_TRIVIAL(error) << "Error " << status;
          break;
      }
      if (status!=0) {
        assert(status==0);
      }
      t=gsl_root_fsolver_root(solver_);
      double t_low=gsl_root_fsolver_x_lower(solver_);
      double t_high=gsl_root_fsolver_x_upper(solver_);
      test_status=gsl_root_test_interval(t_low, t_high, resolution, 0);
      ++iter;
    }
    if (test_status!=GSL_SUCCESS) {
      BOOST_LOG_TRIVIAL(error) << "test status " << test_status;
    }
    if (iter==iter_max) {
      BOOST_LOG_TRIVIAL(error)<< "Reached max iteration.";
    }

    return t;
  }
};


#endif // _SEASONAL_INTERVAL_H_
