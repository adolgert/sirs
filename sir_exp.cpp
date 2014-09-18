
#include <tuple>
#include <map>
#include <iostream>
#include <fstream>
#include <limits>
#include <algorithm>
#include <memory>
#include <set>
#include <functional>
#include "stochnet.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/log/core.hpp"
#include "boost/math/constants/constants.hpp"
#include "boost/property_map/property_map.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/program_options.hpp"
#include "boost/bimap/bimap.hpp"
#include "boost/bimap/set_of.hpp"
#include "boost/bimap/multiset_of.hpp"
#include "smv.hpp"
#include "sir_exp.hpp"
#include "seasonal.hpp"

namespace smv=afidd::smv;
using namespace smv;


struct IndividualToken
{
  double time_entered_place;
  int64_t id;
  IndividualToken()=default;
  IndividualToken(int64_t id, double when) : id(id), time_entered_place(when) {}

  inline friend
  std::ostream& operator<<(std::ostream& os, const IndividualToken& it){
    return os << "T";
  }
};


struct SIRPlace
{
  int disease;
  SIRPlace()=default;
  SIRPlace(int d) : disease(d) {}
  friend inline
  bool operator<(const SIRPlace& a, const SIRPlace& b) {
    return LazyLess(a.disease, b.disease);
  }


  friend inline
  bool operator==(const SIRPlace& a, const SIRPlace& b) {
    return a.disease==b.disease;
  }


  friend inline
  std::ostream& operator<<(std::ostream& os, const SIRPlace& cp) {
    return os << '(' << cp.disease << ')';
  }
};


struct SIRTKey
{
  int kind;
  SIRTKey()=default;
  SIRTKey(int k) : kind(k) {}

  friend inline
  bool operator<(const SIRTKey& a, const SIRTKey& b) {
    return LazyLess(a.kind, b.kind);
  }

  friend inline
  bool operator==(const SIRTKey& a, const SIRTKey& b) {
    return a.kind==b.kind;
  }

  friend inline
  std::ostream& operator<<(std::ostream& os, const SIRTKey& cp) {
    return os << '(' << cp.kind << ')';
  }
};


// This is as much of the marking as the transition will see.
using Local=LocalMarking<Uncolored<IndividualToken>>;
// Extra state to add to the system state. Will be passed to transitions.
struct WithParams {
  // Put our parameters here.
  std::map<SIRParam,double> params;
  int64_t token_cnt;
};


// The transition needs to know the local marking and any extra state.
using SIRTransition=ExplicitTransition<Local,RandGen,WithParams>;

using Dist=TransitionDistribution<RandGen>;
using ExpDist=ExponentialDistribution<RandGen>;


template<typename TransitionType, typename RNG>
class CombinedDistribution : public TransitionDistribution<RNG> {
 public:
  CombinedDistribution(TransitionType* parent) : parent_transition_(parent) {}
  virtual ~CombinedDistribution() {}
  virtual double Sample(double current_time, RNG& rng) const {
    return parent_transition_->Sample(current_time, rng);
  };
  virtual double EnablingTime() const { return 0.0; }
  virtual bool BoundedHazard() const { return false; }
  virtual double HazardIntegral(double t0, double t1) const { return 0.0; }
  virtual double ImplicitHazardIntegral(double xa, double t0) const {
    return 0.0;
  }
 private:
  TransitionType* parent_transition_;
};




// Now make specific transitions.
class Infect : public SIRTransition
{
  using TokenId=int64_t;
  PropagateCompetingProcesses<TokenId,RandGen> propagator_;
  std::tuple<TokenId,double> sample_;
  // This object must track what tokens have been changed.
  using Time=double;
  using InOrOut=bool;
  using Indicator=boost::bimaps::bimap<boost::bimaps::set_of<TokenId>,
    boost::bimaps::multiset_of<InOrOut>, boost::bimaps::with_info<Time>>;
  Indicator ids_;
  bool seen_flag_;

public:
  Infect() : seen_flag_(false) {}
  virtual ~Infect() {}

  virtual std::pair<bool, std::unique_ptr<Dist>>
  Enabled(const UserState& s, const Local& lm,
    double te, double t0, RandGen& rng) override {
    // If these are just size_t, then the rate calculation overflows.
    int64_t S=lm.template Length<0>(0);
    int64_t I=lm.template Length<0>(1);
    int64_t R=lm.template Length<0>(2);

    using IndEntry=Indicator::value_type;

    if (S>0 && I>0) {
      SMVLOG(BOOST_LOG_TRIVIAL(trace)<<"Infect::Enabled infect S "<<S<<" I "<<I);
      // Remove the I factor because this is the rate for each I.
      double rate=S*s.params.at(SIRParam::Beta0)/(S+I+R);
      bool found;
      int token_count;
      // Keep track of which tokens were added, unmodified, modified.
      std::tie(token_count, found)=lm.Get<0>(1,
        [&] (const std::vector<IndividualToken>& tokens)->int {
          for (const auto& t : tokens) {
            auto indicator=ids_.left.find(t.time_entered_place);
            SMVLOG(BOOST_LOG_TRIVIAL(trace)<<"Infect::Enabled token "<<t.id
              <<" time "<<t.time_entered_place<<" found "
              <<(indicator==ids_.left.end()));
            if (indicator==ids_.left.end()) {
              auto dist=std::unique_ptr<Dist>(new ExpDist(rate, te));
              propagator_.Enable(t.id, dist, te, false, rng);
              ids_.insert(IndEntry{t.id, !seen_flag_, t.time_entered_place});
            } else if (indicator->info==t.time_entered_place) {
              ids_.left.erase(indicator); // Because entry is const.
              ids_.insert(IndEntry{t.id, !seen_flag_, t.time_entered_place});
            } else {
              auto dist=std::unique_ptr<Dist>(new ExpDist(rate, te));
              propagator_.Enable(t.id, dist, te, false, rng);
              ids_.left.erase(indicator);
              ids_.insert(IndEntry{t.id, !seen_flag_, t.time_entered_place});
              indicator->info=t.time_entered_place;
            }
          }
          return 0;
        });
      auto erase_it=ids_.right.equal_range(seen_flag_);
      auto convert=erase_it.first;
      for ( ; convert!=erase_it.second; ++convert) {
        SMVLOG(BOOST_LOG_TRIVIAL(trace)<<"Infect::Enabled disable "<<convert->first);
        propagator_.Disable(convert->first, te);
      }
      ids_.right.erase(erase_it.first, erase_it.second);
      seen_flag_=!seen_flag_;
      return {true, std::unique_ptr<Dist>(
        new CombinedDistribution<Infect,RandGen>(this))};
    } else {
      SMVLOG(BOOST_LOG_TRIVIAL(trace)<<"infection disable");
      return {false, std::unique_ptr<Dist>(nullptr)};
    }
  }

  double Sample(double now, RandGen& rng) {
    sample_=propagator_.Next(now, rng);
    SMVLOG(BOOST_LOG_TRIVIAL(trace)<<"Infect::Sample id "<<std::get<0>(sample_)
      <<" time "<<std::get<1>(sample_));
    return std::get<1>(sample_);
  }

  virtual void Fire(UserState& s, Local& lm, double t0,
      RandGen& rng) override {
    SMVLOG(BOOST_LOG_TRIVIAL(trace) << "Fire infection " << lm);
    // s0 i1 r2 i3 r4
    SMVLOG(BOOST_LOG_TRIVIAL(trace)<<"Infect::Fire id "<<std::get<0>(sample_)
      <<" time "<<std::get<1>(sample_));
    auto modify_func=[t0](IndividualToken& t) {
        t.time_entered_place=t0;
      };
    lm.template Move<0,0,decltype(modify_func)>(0, 3, 1, modify_func);
  }
};



// Now make specific transitions.
class InfectExact : public SIRTransition
{
  virtual std::pair<bool, std::unique_ptr<Dist>>
  Enabled(const UserState& s, const Local& lm,
    double te, double t0, RandGen& rng) override {
    // If these are just size_t, then the rate calculation overflows.
    int64_t S=lm.template Length<0>(0);
    int64_t I=lm.template Length<0>(1);
    int64_t R=lm.template Length<0>(2);
    if (S>0 && I>0) {
      return {true, std::unique_ptr<SeasonalBeta<RandGen>>(
        new SeasonalBeta<RandGen>(S*I*s.params.at(SIRParam::Beta0)/(S+I+R),
          s.params.at(SIRParam::Beta1), s.params.at(SIRParam::SeasonalPhase),
          te))};
    } else {
      return {false, std::unique_ptr<Dist>(nullptr)};
    }
  }

  virtual void Fire(UserState& s, Local& lm, double t0,
      RandGen& rng) override {
    SMVLOG(BOOST_LOG_TRIVIAL(trace) << "Fire infection " << lm);
    lm.template Move<0,0>(0, 3, 1);
  }
};



// Now make specific transitions.
class Recover : public SIRTransition
{
  virtual std::pair<bool, std::unique_ptr<Dist>>
  Enabled(const UserState& s, const Local& lm,
    double te, double t0, RandGen& rng) override {
    int64_t I=lm.template Length<0>(0);
    if (I>0) {
      double rate=I*s.params.at(SIRParam::Gamma);
      SMVLOG(BOOST_LOG_TRIVIAL(trace)<<"recover rate "<< rate);
      return {true, std::unique_ptr<ExpDist>(
        new ExpDist(rate, te))};
    } else {
      SMVLOG(BOOST_LOG_TRIVIAL(trace)<<"recover disable");
      return {false, std::unique_ptr<Dist>(nullptr)};
    }
  }

  virtual void Fire(UserState& s, Local& lm, double t0,
      RandGen& rng) override {
    SMVLOG(BOOST_LOG_TRIVIAL(trace) << "Fire recover " << lm);
    lm.template Move<0, 0>(0, 1, 1);
  }
};




// Now make specific transitions.
class Wane : public SIRTransition
{
  virtual std::pair<bool, std::unique_ptr<Dist>>
  Enabled(const UserState& s, const Local& lm,
    double te, double t0, RandGen& rng) override {
    int64_t S=lm.template Length<0>(0);
    double rate=S*s.params.at(SIRParam::Wane);
    if (S>0 && rate>0) {
      SMVLOG(BOOST_LOG_TRIVIAL(trace)<<"wane rate "<< rate);
      return {true, std::unique_ptr<ExpDist>(
        new ExpDist(rate, te))};
    } else {
      SMVLOG(BOOST_LOG_TRIVIAL(trace)<<"wane disable");
      return {false, std::unique_ptr<Dist>(nullptr)};
    }
  }

  virtual void Fire(UserState& s, Local& lm, double t0,
      RandGen& rng) override {
    SMVLOG(BOOST_LOG_TRIVIAL(trace) << "Fire wane " << lm);
    lm.template Move<0, 0>(0, 1, 1);
  }
};


// Now make specific transitions.
class Birth : public SIRTransition
{
  virtual std::pair<bool, std::unique_ptr<Dist>>
  Enabled(const UserState& s, const Local& lm,
    double te, double t0, RandGen& rng) override {
    if (s.params.at(SIRParam::Birth)>0) {
      return {true, std::unique_ptr<ExpDist>(
        new ExpDist(s.params.at(SIRParam::Birth), te))};
    } else {
      return {false, std::unique_ptr<Dist>(nullptr)};
    }
  }

  virtual void Fire(UserState& s, Local& lm, double t0,
      RandGen& rng) override {
    SMVLOG(BOOST_LOG_TRIVIAL(trace) << "Fire birth " << lm);
    lm.template Add<0>(1, IndividualToken{s.token_cnt++, t0});
  }
};



// Now make specific transitions.
class Death : public SIRTransition
{
  virtual std::pair<bool, std::unique_ptr<Dist>>
  Enabled(const UserState& s, const Local& lm,
    double te, double t0, RandGen& rng) override {
    int64_t SIR=lm.template Length<0>(0);
    if (SIR>0 && s.params.at(SIRParam::Mu)>0) {
      return {true, std::unique_ptr<ExpDist>(
        new ExpDist(SIR*s.params.at(SIRParam::Mu), te))};
    } else {
      return {false, std::unique_ptr<Dist>(nullptr)};
    }
  }

  virtual void Fire(UserState& s, Local& lm, double t0,
      RandGen& rng) override {
    SMVLOG(BOOST_LOG_TRIVIAL(trace) << "Fire death " << lm);
    lm.template Remove<0>(0, 1, rng);
  }
};




// The GSPN itself.
using SIRGSPN=
    ExplicitTransitions<SIRPlace, SIRTKey, Local, RandGen, WithParams>;

/*! SIR infection on an all-to-all graph of uncolored tokens.
 */
SIRGSPN
BuildSystem(int64_t individual_cnt, bool exactbeta)
{
  BuildGraph<SIRGSPN> bg;
  using Edge=BuildGraph<SIRGSPN>::PlaceEdge;

  enum { s, i, r };

  for (int place : std::vector<int>{s, i, r}) {
    bg.AddPlace({place}, 0);
  }

  enum { infect, recover, wane, birth, deaths, deathi, deathr };

  if (exactbeta) {
    BOOST_LOG_TRIVIAL(info)<<"Using exact seasonal infection";
    bg.AddTransition({infect},
      {Edge{{s}, -1}, Edge{{i}, -1}, Edge{{r}, -1}, Edge{{i}, 2}, Edge{{r}, 1}},
      std::unique_ptr<SIRTransition>(new InfectExact())
      );
  } else {
    BOOST_LOG_TRIVIAL(info)<<"Using piecewise seasonal infection";
    bg.AddTransition({infect},
      {Edge{{s}, -1}, Edge{{i}, -1}, Edge{{r}, -1}, Edge{{i}, 2}, Edge{{r}, 1}},
      std::unique_ptr<SIRTransition>(new Infect())
      );
  }

  bg.AddTransition({recover},
    {Edge{{i}, -1}, Edge{{r}, 1}},
    std::unique_ptr<SIRTransition>(new Recover())
    );

  bg.AddTransition({wane},
    {Edge{{r}, -1}, Edge{{s}, 1}},
    std::unique_ptr<SIRTransition>(new Wane())
    );

  bg.AddTransition({birth},
    {Edge{{s}, -1}, Edge{{s}, 2}},
    std::unique_ptr<SIRTransition>(new Birth())
    );

  bg.AddTransition({deaths},
    {Edge{{s}, -1}, Edge{{s}, 0}},
    std::unique_ptr<SIRTransition>(new Death())
    );

  bg.AddTransition({deathi},
    {Edge{{i}, -1}, Edge{{i}, 0}},
    std::unique_ptr<SIRTransition>(new Death())
    );

  bg.AddTransition({deathr},
    {Edge{{r}, -1}, Edge{{r}, 0}},
    std::unique_ptr<SIRTransition>(new Death())
    );

  // std::move the transitions because they contain unique_ptr.
  return std::move(bg.Build());
}


/*!
 * Given a vector of checkpoint times, for the state of the system
 * at each of those times, count the number of
 * susceptibles and infecteds at that time. Form into two vectors,
 * one for susceptibles, one for infecteds. Then repeat the whole
 * simulation 10^4 times. Return these vectors.
 */
template<typename SIRState>
struct SIROutput
{
  std::vector<int64_t> places_;
  std::map<int64_t,int> transitions_;
  using StateArray=std::array<int64_t,3>;
  double max_time_;
  int64_t max_count_;
  TrajectoryObserver& observer_;
  std::array<int64_t,3> sir_;

  SIROutput(double max_time, int64_t max_count,
    const std::vector<int64_t>& sir_places,
    const std::map<int64_t,int>& sir_transitions,
    TrajectoryObserver& observer)
  : places_{sir_places}, max_time_(max_time), max_count_(max_count),
    observer_(observer), transitions_(sir_transitions) {
    for (auto& t : transitions_) {
      BOOST_LOG_TRIVIAL(debug)<<"gspn transition "<<t.first<<"="<<t.second;
    }
    for (auto& p: places_) {
      BOOST_LOG_TRIVIAL(debug)<<"gspn place "<<p;
    }
  };

  int64_t step_cnt{0};

  void operator()(const SIRState& state) {
    int64_t S=Length<0>(state.marking, places_[0]);
    int64_t I=Length<0>(state.marking, places_[1]);
    int64_t R=Length<0>(state.marking, places_[2]);
    SMVLOG(BOOST_LOG_TRIVIAL(trace)<<"last transition "<<state.last_transition);
    SMVLOG(BOOST_LOG_TRIVIAL(trace)<<"S="<<S<<" I="<<I<<" R="<<R);

    if (step_cnt>0) {
      switch (transitions_[state.last_transition]) {
        case 0: // infect
          assert(S+I+R==sir_[0]+sir_[1]+sir_[2]);
          assert(sir_[0]-S==1);
          assert(I-sir_[1]==1);
          assert(R==sir_[2]);
          break;
        case 1: // recover
          assert(S+I+R==sir_[0]+sir_[1]+sir_[2]);
          assert(sir_[1]-I==1);
          assert(R-sir_[2]==1);
          assert(S==sir_[0]);
          break;
        case 2: // wane
          assert(S+I+R==sir_[0]+sir_[1]+sir_[2]);
          assert(sir_[2]==R+1);
          assert(sir_[0]==S-1);
          assert(I==sir_[1]);
          break;
        case 3: // birth
          assert(I==sir_[1]);
          assert(R==sir_[2]);
          assert(S==sir_[0]+1);
          break;
        case 4: // death s
          assert(I==sir_[1]);
          assert(R==sir_[2]);
          assert(S==sir_[0]-1);
          break;
        case 5: // death i
          assert(S==sir_[0]);
          assert(R==sir_[2]);
          assert(I==sir_[1]-1);
          break;
        case 6: // death r
          assert(S==sir_[0]);
          assert(I==sir_[1]);
          assert(R==sir_[2]-1);
          break;
        default:
          assert(transitions_[state.last_transition]<7);
          assert(transitions_[state.last_transition]>=0);
          break;
      }
    } else {
      ; // This is the first step.
    }

    ++step_cnt;
    observer_.Step({S, I, R, state.CurrentTime()});
    sir_={S, I, R};
  }

  void final(const SIRState& state) {
    BOOST_LOG_TRIVIAL(info) << "Took "<< step_cnt << " transitions.";
  }
};



int64_t SIR_run(double end_time, const std::vector<int64_t>& sir_cnt,
    const std::vector<Parameter>& parameters, TrajectoryObserver& observer,
    RandGen& rng, bool infect_exact)
{
  int64_t individual_cnt=std::accumulate(sir_cnt.begin(), sir_cnt.end(),
    int64_t{0});
  auto gspn=BuildSystem(individual_cnt, infect_exact);

  // Marking of the net.
  static_assert(std::is_same<int64_t,SIRGSPN::PlaceKey>::value,
    "The GSPN's internal place type is int64_t.");
  using Mark=Marking<SIRGSPN::PlaceKey, Uncolored<IndividualToken>>;
  using SIRState=GSPNState<Mark,SIRGSPN::TransitionKey,WithParams>;

  SIRState state;
  for (auto& cp : parameters) {
    state.user.params[cp.kind]=cp.value;
  }

  std::vector<int64_t> sir_places;
  for (int place_idx=0; place_idx<3; ++place_idx) {
    auto place=gspn.PlaceVertex({place_idx});
    sir_places.push_back(place);
  }

  int64_t token_id=0;
  for (int64_t sir_idx=0; sir_idx<3; ++sir_idx) {
    for (int64_t sus_idx=0; sus_idx<sir_cnt[sir_idx]; ++sus_idx) {
      Add<0>(state.marking, sir_places[sir_idx],
        IndividualToken{token_id++, 0.0});
    }
  }
  state.user.token_cnt=token_id;

  //using Propagator=PropagateCompetingProcesses<int64_t,RandGen>;
  using Propagator=NonHomogeneousPoissonProcesses<int64_t,RandGen>;
  PropagateCompetingProcesses<int64_t,RandGen> simple;
  Propagator competing;
  using Dynamics=StochasticDynamics<SIRGSPN,SIRState,RandGen>;
  Dynamics dynamics(gspn, {&competing, &simple});

  BOOST_LOG_TRIVIAL(debug) << state.marking;

  std::map<int64_t,int> trans;
  for (size_t t_idx=0; t_idx<7; ++t_idx) {
    trans[gspn.TransitionVertex(t_idx)]=t_idx;
  }

  SIROutput<SIRState> output_function(end_time, individual_cnt*2,
    sir_places, trans, observer);

  dynamics.Initialize(&state, &rng);

  bool running=true;
  auto nothing=[](SIRState&)->void {};
  double last_time=state.CurrentTime();
  while (running && state.CurrentTime()<end_time) {
    running=dynamics(state);
    if (running) {
      double new_time=state.CurrentTime();
      if (new_time-last_time<-1e-12) {
        BOOST_LOG_TRIVIAL(warning) << "last time "<<last_time <<" "
          << " new_time "<<new_time;
      }
      last_time=new_time;
      output_function(state);
    }
  }
  if (running) {
    BOOST_LOG_TRIVIAL(info)<<"Reached end time "<<state.CurrentTime();
  } else {
    BOOST_LOG_TRIVIAL(info)<<"No transitions left to fire at time "<<last_time;
  }
  output_function.final(state);
  return 0;
}

