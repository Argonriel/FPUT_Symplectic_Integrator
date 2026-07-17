# Makefile for the Yoshida FPUT solver and its self-test driver.
#
# The production solver still compiles with the plain command documented in the
# source header; this Makefile additionally stamps the current Git short hash
# into the binary (metadata field "# SolverVersion:") for reproducibility.
#
#   make            # build fput_yoshida (production) + fput_selftest (tests)
#   make fput_yoshida
#   make fput_selftest
#   make clean
#
# Override the compiler/flags if needed, e.g.:  make CXX=g++-14

CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra

# Git provenance for production data. GIT_COMMIT is the short hash; GIT_DIRTY is
# 1 if any TRACKED file is modified/staged (untracked scratch files are ignored,
# since they do not affect the build). A dirty worktree must not produce
# production data -- the header records SolverGitDirty so this is auditable.
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
GIT_DIRTY  := $(shell test -z "$$(git status --porcelain --untracked-files=no 2>/dev/null)" && echo 0 || echo 1)
VERSION_DEF := -DFPUT_GIT_COMMIT=\"$(GIT_COMMIT)\" -DFPUT_GIT_DIRTY=$(GIT_DIRTY)

HEADERS := fput_physics.hpp fput_checkpoint.hpp

.PHONY: all clean
all: fput_yoshida fput_selftest

fput_yoshida: FPUT_yoshida_solver.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(VERSION_DEF) -o $@ FPUT_yoshida_solver.cpp

# The self-test driver does not need -march=native; keep it simple and portable.
fput_selftest: fput_selftest.cpp $(HEADERS)
	$(CXX) -O2 -std=c++17 -Wall -Wextra -o $@ fput_selftest.cpp

clean:
	rm -f fput_yoshida fput_selftest
