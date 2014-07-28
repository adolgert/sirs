#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>
#include "gsl/gsl_errno.h"
#include "gsl/gsl_math.h"
#include "gsl/gsl_roots.h"
#include "seasonal.hpp"
#include "smv.hpp"



namespace seasonal {
double seasonal_hazard(double t, void* params) {
  SeasonalParams* p = static_cast<SeasonalParams*>(params);
  //std::cout << "beta2 "<<p->beta <<" pti "<<p->ti<<" px "<<p->X<<std::endl;
  return t + p->beta*std::sin(2*M_PI*(t - p->ti)) - p->X;
}


double seasonal_hazard_df(double t, void* params) {
  SeasonalParams* p = static_cast<SeasonalParams*>(params);
  return 1 + 2*M_PI*p->beta*std::cos(2*M_PI*(t - p->ti));
}


void seasonal_hazard_fdf(double t, void* params,
    double* f, double* df) {
  SeasonalParams* p = static_cast<SeasonalParams*>(params);
  *f=t + p->beta*std::sin(2*M_PI*(t - p->ti)) - p->X;
  *df=1 + 2*M_PI*p->beta*std::cos(2*M_PI*(t - p->ti));
}
}



double TestSeasonal(double b1) {
  std::mt19937 rng;

  int ti_cnt=100;
  int t0_cnt=100;
  int t_cnt=100;
  int b_cnt=100;
  double tol=0;
  double eps=1e-14;
  for (int ti_idx=0; ti_idx<ti_cnt; ++ti_idx) {
    double ti=ti_idx/(double) ti_cnt;
    for (int t0_idx=0; t0_idx<t0_cnt; ++t0_idx) {
      double t0=4*t0_idx/(double) t0_cnt;
      for (int b_idx=0; b_idx<b_cnt; ++b_idx) {
        double b=0.1+20*b_idx/(double) b_cnt;
        for (int t_idx=0; t_idx<t_cnt; ++t_idx) {
          double t=t0+20*t_idx/(double) t_cnt;
          double U=b*((t-t0)+(b1/(2*M_PI))*
              (std::sin(2*M_PI*(t-ti)) - std::sin(2*M_PI*(t0-ti))));
          double X=U/b + t0 + (b1/(2*M_PI))*std::sin(2*M_PI*(t0-ti));
          double dX=b1/(2*M_PI);
          if (t<X-dX-eps) {
            BOOST_LOG_TRIVIAL(error) << "t outside bounds " << t <<
              " Xmin" << X-dX << " Xmax " << X+dX << " " << t-(X-dX);
          }
          if (t>X+dX+eps) {
            BOOST_LOG_TRIVIAL(error) << "t outside bounds " << t <<
              " Xmin" << X-dX << " Xmax " << X+dX << " " <<
              t-(X+dX);
          }
          assert(U>=0);
          auto s=SeasonalBeta<std::mt19937>(b, b1, ti, 0.0);
          double pred1=s.ImplicitHazardIntegral(U, t0);
          //std::cout << predicted << " " << t << " " << X-dX << " " << X+dX <<" "
          //    <<t0 << " " << ti << " " << b << " " << b1 << " " << U << std::endl; 
          tol=std::max(tol, std::abs(pred1-t));
        }
      }
    }
  }
  return tol;
}
