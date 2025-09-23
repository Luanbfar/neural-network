# 1. Compiler and Flags
# ------------------------------------
# Use g++ as the C++ compiler.
CXX = g++
# Compiler flags:
# -std=c++17: Use the C++17 standard.
# -Iinclude:  Look for header files in the 'include' directory.
# -Wall:      Enable all standard warnings.
# -Wextra:    Enable extra (non-standard) warnings.
# -g:         Include debugging information.
CXXFLAGS = -std=c++17 -Iinclude -Wall -Wextra -g

# 2. Project Structure
# ------------------------------------
# The name of the final executable.
TARGET = neural_network
# The directory for compiled object files and the final executable.
BUILD_DIR = build
# The directory containing your source (.cpp) files.
SRC_DIR = src

# 3. Automatic File Detection
# ------------------------------------
# Automatically find all .cpp files in the SRC_DIR.
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
# Generate a list of object (.o) files to be placed in the BUILD_DIR.
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# 4. Build Rules
# ------------------------------------
# The default target, 'all', depends on the final executable.
# Running 'make' will build this target.
all: $(BUILD_DIR)/$(TARGET)

# Rule to link all object files into the final executable.
$(BUILD_DIR)/$(TARGET): $(OBJS)
	@echo "==> Linking executable: $@"
	$(CXX) $(CXXFLAGS) -o $@ $^

# Pattern rule to compile a .cpp file into a .o file.
# This rule creates the build directory if it doesn't exist,
# then compiles the source file.
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	@echo "==> Compiling: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 5. Utility Rules
# ------------------------------------
# The 'clean' rule removes the build directory and its contents.
clean:
	@echo "==> Cleaning project..."
	rm -rf $(BUILD_DIR)

# The 'run' rule compiles and then executes the program.
run: all
	@echo "==> Running project..."
	./$(BUILD_DIR)/$(TARGET)

# Declare targets that are not files.
.PHONY: all clean run