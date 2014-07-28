# Requires:
#   BOOST libraries: boost.org
#   Gnu scientific libraries (GSL): https://www.gnu.org/software/gsl
#   HDF5 library: http://www.hdfgroup.org/HDF5/
#   Semi-Markov library: https://github.com/afidd/Semi-Markov
#   mcrand library: https://github.com/afidd/mcrand
#
# mcrand is a pain. It exists so that runs with different seeds will
# be reliably random, but it depends on RNGSSELIB by Barash and Shchur,
# which you have to retrieve from Elsevier to build it. If you feel lucky with
# the Boost random generator, then go to sir_exp.hpp and change
# RandGen to use that one, and you can take mcrand out of this Makefile.

BOOST=/usr/local/boost_1_54_0mt
# Different Boost installations have different suffixes.
# If there is no suffix, use "BOOSTVARIANT=".
BOOSTVARIANT=-mt
SEMIMARKOV=/usr/local/include/semimarkov-0.1
HDF5=/usr/local/hdf5-1.8.11
MCRAND=/home/ajd27/Documents/mcrand

CXX=g++
# -DSMVHIDELOG -pg
OPT=-g -O2 -DSMVHIDELOG
INCLUDES=-I$(SEMIMARKOV) -I. -I$(BOOST)/include -I$(HDF5)/include -I$(MCRAND)/include
LIBS=-L$(BOOST)/lib -L$(HDF5)/lib -L$(MCRAND)/lib \
    -lboost_unit_test_framework$(BOOSTVARIANT) \
	-lboost_log_setup$(BOOSTVARIANT) -lboost_log$(BOOSTVARIANT) \
	-lboost_chrono$(BOOSTVARIANT) -lboost_thread$(BOOSTVARIANT) \
	-lboost_date_time$(BOOSTVARIANT) -lboost_filesystem$(BOOSTVARIANT) \
	-lboost_program_options$(BOOSTVARIANT) -lboost_random$(BOOSTVARIANT) \
	-lboost_system$(BOOSTVARIANT) -lhdf5 -lhdf5_hl -lgsl -lgslcblas \
	-lmcrand -lpthread


sirexp: sir_exp.o main.o seasonal.o hdf_file.o
	g++ $(OPT) -fPIC -o sirexp sir_exp.o main.o seasonal.o hdf_file.o $(LIBS)

sir_exp.o: sir_exp.cpp sir_exp.hpp seasonal.hpp
	g++ sir_exp.cpp -DHAVE_CONFIG_H -std=c++11 -fPIC $(INCLUDES) $(OPT) \
	-c -o sir_exp.o

seasonal.o: seasonal.cpp seasonal.hpp
	g++ seasonal.cpp -DHAVE_CONFIG_H -std=c++11 -fPIC $(INCLUDES) $(OPT) \
	-c -o seasonal.o

hdf_file.o: hdf_file.cpp hdf_file.hpp sir_exp.hpp
	g++ hdf_file.cpp -DHAVE_CONFIG_H -std=c++11 -fPIC $(INCLUDES) $(OPT) \
	-c -o hdf_file.o

main.o: main.cpp sirdemo_version.hpp sir_exp.hpp
	g++ main.cpp -DHAVE_CONFIG_H -std=c++11 -fPIC $(INCLUDES) $(OPT) \
	-c -o main.o

sirdemo_version.hpp: Makefile
	python getgit.py

clean:
	rm -f *.o sirexp sirdemo_version.hpp
