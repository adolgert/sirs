#ifndef _HDF_FILE_H_
#define _HDF_FILE_H_ 1

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <array>
#include "boost/program_options.hpp"
#include "sir_exp.hpp"

class HDFFile {
 public:
  using TrajectoryType=std::vector<TrajectoryEntry>;
  explicit HDFFile(const std::string& filename);
  HDFFile(const HDFFile& o);
  HDFFile& operator=(const HDFFile&);
  ~HDFFile();
  bool Open(bool truncate=true);
  bool Close();
  bool SaveTrajectory(const std::vector<Parameter>& params,
    int seed, int idx, const TrajectoryType& trajectory) const;
  bool WriteExecutableData(const std::map<std::string,std::string>& compile,
    boost::program_options::basic_parsed_options<char>& cmdline,
    const std::vector<int64_t>& initial_values) const;
 private:
  class impl;
  std::shared_ptr<impl> pimpl;
};


#endif

