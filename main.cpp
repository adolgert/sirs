#include <string>
#include <sstream>
#include "boost/program_options.hpp"
#include "smv.hpp"
#include "sir_exp.hpp"
#include "hdf_file.hpp"
#include "seasonal.hpp"
#include "ensemble.hpp"
#include "sirdemo_version.hpp"




/*! Save the whole trajectory.
 */
class TrajectorySave : public TrajectoryObserver
{
  std::vector<TrajectoryEntry> trajectory_;
 public:
  virtual void Step(TrajectoryEntry sirt) override {
    trajectory_.emplace_back(sirt);
  }
  virtual const std::vector<TrajectoryEntry>& Trajectory() const {
    return trajectory_; }
};



/*! Save the trajectory every time any of SIR change by a percent.
 */
class PercentTrajectorySave : public TrajectoryObserver
{
  int64_t step_{0};
  int64_t threshhold_{0};
  double percent_{0.0001};

  TrajectoryEntry last_{0,0,0,0};
  std::vector<TrajectoryEntry> trajectory_;
 public:
  PercentTrajectorySave() {}

  virtual void Step(TrajectoryEntry sirt) override {
    if (0==step_) {
      last_=sirt;
      threshhold_=std::floor(percent_*(sirt.s+sirt.i+sirt.r));
      trajectory_.emplace_back(sirt);
    } else {
      bool ps=std::abs(sirt.s-last_.s)>threshhold_;
      bool pi=std::abs(sirt.i-last_.i)>threshhold_;
      bool pr=std::abs(sirt.r-last_.r)>threshhold_;
      if (ps||pi||pr) {
        trajectory_.emplace_back(sirt);
        last_=sirt;
      }
    }
    ++step_;
  }
  virtual const std::vector<TrajectoryEntry>& Trajectory() const {
    return trajectory_; }
};




int main(int argc, char *argv[]) {
  namespace po=boost::program_options;
  po::options_description desc("Well-mixed SIR with demographics.");
  int64_t individual_cnt=100000;
  int64_t infected_cnt=individual_cnt*0.1;
  int64_t recovered_cnt=individual_cnt*0.8;

  int run_cnt=1;
  size_t rand_seed=1;
  // Time is in years.
  std::vector<Parameter> parameters;
  parameters.emplace_back(Parameter{SIRParam::Wane, "wane", 0.5,
    "waning immunity rate"});
  parameters.emplace_back(Parameter{SIRParam::Beta0, "beta0", 400,
    "main infection rate"});
  parameters.emplace_back(Parameter{SIRParam::Beta1, "beta1", 0.6,
    "seasonality ratio"});
  parameters.emplace_back(Parameter{SIRParam::SeasonalPhase, "phase", 0,
    "seasonality phase start between (0,1]"});
  parameters.emplace_back(Parameter{SIRParam::Gamma, "gamma", 365/14.0,
    "recovery rate"});
  parameters.emplace_back(Parameter{SIRParam::Birth, "birth", 1/70.0,
    "crude rate, before multiplying by number of individuals"});
  parameters.emplace_back(Parameter{SIRParam::Mu, "mu", 1/70.0,
    "death rate"});
  double end_time=30.0;
  bool exacttraj=true;
  bool exactinfect=false;
  int thread_cnt=1;
  std::string log_level;
  std::string data_file("sirexp.h5");
  bool save_file=false;
  std::string translation_file;
  bool test=false;

  desc.add_options()
    ("help", "show help message")
    ("threadcnt,j",
      po::value<int>(&thread_cnt)->default_value(thread_cnt),
      "number of threads")
    ("runcnt",
      po::value<int>(&run_cnt)->default_value(run_cnt),
      "number of runs")
    ("size,s",
      po::value<int64_t>(&individual_cnt)->default_value(individual_cnt),
      "size of the population")
    ("infected,i",
      po::value<int64_t>(&infected_cnt),
      "number of infected")
    ("recovered,r",
      po::value<int64_t>(&recovered_cnt),
      "number of recovered")
    ("seed",
      po::value<size_t>(&rand_seed)->default_value(rand_seed),
      "seed for random number generator")
    ("endtime",
      po::value<double>(&end_time)->default_value(end_time),
      "how many years to run")
    ("exacttraj",
      po::value<bool>(&exacttraj)->default_value(exacttraj),
      "save trajectory only when it changes by a certain amount")
    ("exactinfect",
      po::value<bool>(&exactinfect)->default_value(exactinfect),
      "set true to use exact distribution for seasonal infection")
    ("datafile",
      po::value<std::string>(&data_file)->default_value(data_file),
      "Write to this data file.")
    ("save",
      po::value<bool>(&save_file)->default_value(save_file),
      "Add data to file instead of erasing it with new data.")
    ("loglevel", po::value<std::string>(&log_level)->default_value("info"),
      "Set the logging level to trace, debug, info, warning, error, or fatal.")
    ("translate",
      po::value<std::string>(&translation_file)->default_value(""),
      "write file relating place ids to internal ids")
    ("info", "show provenance of program")
    ;

  for (auto& p : parameters) {
    desc.add_options()(p.name.c_str(),
      po::value<double>(&p.value)->default_value(p.value),
      p.description.c_str());
  }

  po::variables_map vm;
  auto parsed_options=po::parse_command_line(argc, argv, desc);
  po::store(parsed_options, vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  std::map<std::string,std::string> compile_info {
    {"VERSION", VERSION}, {"COMPILETIME", COMPILETIME},
    {"CONFIG", CFG}
  };
  if (vm.count("info")) {
    for (auto& kv : compile_info) {
      std::cout << kv.second << "\n\n";
    }
    return 0;
  }

  afidd::LogInit(log_level);

  if (test) {
    double tolerance=TestSeasonal(0.6);
    std::cout << "Seasonal tolerance " << tolerance << std::endl;
  }

  std::map<SIRParam,double*> params;
  for (auto& pm : parameters) {
    params[pm.kind]=&pm.value;
  }

  // Birthrate is not frequency-dependent. It scales differently
  // which creates a fixed point in the phase plane.
  (*params[SIRParam::Birth])*=individual_cnt;

  if (std::abs(*params[SIRParam::Beta1])>1) {
    std::cout << "beta1 should be fractional, between 0 and 1: beta1=" <<
      *params[SIRParam::Beta1] << std::endl;
    return -4;
  }

  if ((vm.count("infected") && !vm.count("recovered")) ||
      (!vm.count("infected") && vm.count("recovered"))) {
    std::cout << "You have so set the total and I and R, not just some of them."
      << std::endl;
    return -3;
  } else if (!vm.count("infected") && !vm.count("recovered")) {
    double b=*params[SIRParam::Beta0];
    double m=*params[SIRParam::Mu];
    double g=*params[SIRParam::Gamma];
    double B=*params[SIRParam::Birth];
    // Long-time averages for fixed forcing for ODE model.
    int64_t susceptible_start=std::floor((m+g)*individual_cnt/b);
    infected_cnt=std::floor(individual_cnt*(b-m-g)*m/(b*(m+g)));
    recovered_cnt=individual_cnt-(susceptible_start+infected_cnt);
  }

  int64_t susceptible_cnt=individual_cnt-(infected_cnt+recovered_cnt);
  assert(susceptible_cnt>0);
  if (susceptible_cnt<0) {
    BOOST_LOG_TRIVIAL(error)<<"Number of susceptibles is "<<susceptible_cnt;
    return -2;
  }
  std::vector<int64_t> sir_init{susceptible_cnt, infected_cnt, recovered_cnt};
  BOOST_LOG_TRIVIAL(info)<<"Starting with sir="<<sir_init[0]<<" "<<sir_init[1]
    <<" "<<sir_init[2];

  for (auto& showp : parameters) {
    BOOST_LOG_TRIVIAL(info)<<showp.name<<" "<<showp.value;
  }

  HDFFile file(data_file);
  if (!file.Open(!save_file)) {
    BOOST_LOG_TRIVIAL(error)<<"could not open output file: "<<data_file;
    return -1;
  }
  file.WriteExecutableData(compile_info, parsed_options, sir_init);

  auto runnable=[=](RandGen& rng, size_t single_seed, size_t idx)->void {
    std::shared_ptr<TrajectoryObserver> observer=0;
    if (exacttraj) {
      observer=std::make_shared<TrajectorySave>();
    } else {
      observer=std::make_shared<PercentTrajectorySave>();
    }

    SIR_run(end_time, sir_init, parameters, *observer, rng, exactinfect);
    file.SaveTrajectory(parameters, single_seed, idx, observer->Trajectory());
  };

  afidd::smv::Ensemble<decltype(runnable),RandGen> ensemble(runnable, thread_cnt,
      run_cnt, rand_seed);
  ensemble.Run();
  BOOST_LOG_TRIVIAL(debug)<<"Finished running ensemble.";

  file.Close();

  return 0;
}
