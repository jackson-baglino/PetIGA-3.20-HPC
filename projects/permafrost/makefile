TARGETS = \
    permafrost \

# Default target: build the executable
all: $(TARGETS)
.PHONY: all

###############################################################################
# Build Mode: Debug vs. Optimized
#
# Use "make debug" to compile with debugging flags.
# Otherwise, the default build mode is optimized.
###############################################################################
ifeq ($(BUILD),debug)
  # Debug mode: maximum debugging information, no optimization.
  CFLAGS = -g3 -O0 -Wall -Wextra -Wno-unused-parameter $(INCLUDES)
  FFLAGS = -g3 -O0 -Wall -Wextra -fcheck=all
else
  # Optimized mode: high optimization for performance.
  CFLAGS = -O3 -Wall -Wextra -Wno-unused-parameter $(INCLUDES)
  FFLAGS = -O3 -Wall -Wextra
endif

###############################################################################
# Compiler and Linker Settings
###############################################################################
# Use the MPI compiler wrapper.
CC = mpicc

# Linker flags: Link against PETSc, PetIGA, and additional dependencies.
LDFLAGS = -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc \
          -L$(PETIGA_DIR)/$(PETSC_ARCH)/lib -lpetiga -lmpi -lmetis -lparmetis -lsuperlu_dist

###############################################################################
# Include Paths
#
# Add paths for PETSc (including private headers), PetIGA, and local headers.
###############################################################################
INCLUDES = -I$(PETSC_DIR)/include \
           -I$(PETSC_DIR)/include/petsc/private \
           -I$(PETIGA_DIR)/include \
           -I./include
CFLAGS += $(INCLUDES)

###############################################################################
# Include PetIGA Configuration Files
#
# These files define additional variables needed for linking:
#  - CLINKER : The linker command to use.
#  - PETIGA_LIB : Additional PetIGA libraries.
#  - PETSC_LIB : Additional PETSc libraries.
###############################################################################
include $(PETIGA_DIR)/lib/petiga/conf/variables
include $(PETIGA_DIR)/lib/petiga/conf/rules
include $(PETIGA_DIR)/lib/petiga/conf/test

###############################################################################
# Source and Object Files
#
# List all .c files in the "src" directory and define corresponding object files in "obj".
###############################################################################
SRCS := $(wildcard src/*.c)
OBJS := $(patsubst src/%.c, obj/%.o, $(SRCS))

###############################################################################
# Build Rules
###############################################################################
# Compile rule: Build an object file from a source file.
obj/%.o: src/%.c
	@mkdir -p obj
	$(CC) $(CFLAGS) -c $< -o $@

# Linking rule: Link all object files to create the executable.
# Uses $(CLINKER) (set by the included configuration files) and appends the extra
# libraries from $(PETIGA_LIB) and $(PETSC_LIB).
$(TARGETS): $(OBJS)
	$(CLINKER) -o $@ $^ $(PETIGA_LIB) $(PETSC_LIB)

###############################################################################
# Clean Rule
###############################################################################
# Remove the executable and all object files in the "obj" directory.
###############################################################################
clean::
	@echo "Cleaning up..."
	@rm -f $(TARGETS)
	@find obj -type f -name '*.o' -delete
	@echo "Clean"

###############################################################################
# Debug Build Target
#
# Use "make debug" to compile with debug flags.
###############################################################################
debug:
	$(MAKE) BUILD=debug all
.PHONY: debug