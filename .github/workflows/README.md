# GitHub Actions Workflows

This directory contains CI/CD workflows for the triton-tvm project.

## Workflows

### 1. CI Workflow (`ci.yml`)

**Triggers:**
- Push to `main`, `master`, or `develop` branches
- Pull requests to `main`, `master`, or `develop` branches
- Manual trigger via workflow_dispatch

**Purpose:**
- Build the triton-tvm plugin with Triton
- Verify the build succeeds
- Run example tests (when CUDA is available)
- Archive build artifacts

**Steps:**
1. Sets up Python 3.10 and system dependencies
2. Installs PyTorch and other Python packages
3. Clones and builds Triton with triton-tvm as a plugin
4. Verifies the plugin loads correctly
5. Runs example tests
6. Uploads build artifacts

### 2. Release Build Workflow (`release.yml`)

**Triggers:**
- Push of version tags (format: `v*.*.*`, e.g., `v1.0.0`)
- Manual trigger with custom version input

**Purpose:**
- Create release packages with source code and build instructions
- Generate GitHub releases with downloadable artifacts

**Steps:**
1. Builds the project similar to CI workflow
2. Creates distribution packages (.tar.gz and .zip)
3. Generates build information file
4. Creates GitHub release with artifacts (for tag pushes)

**Creating a Release:**
```bash
# Tag a new version
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Or trigger manually from GitHub Actions UI
```

### 3. Code Quality Workflow (`lint.yml`)

**Triggers:**
- Push to `main`, `master`, or `develop` branches
- Pull requests to `main`, `master`, or `develop` branches

**Purpose:**
- Check code formatting and style
- Run linters on Python and C++ code

**Checks:**
- Python: black (formatting), isort (import sorting), flake8 (linting)
- C++: clang-format (formatting)

## Usage

### Running CI Locally

To test the build process locally before pushing:

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install build-essential cmake ninja-build ccache clang lld llvm-dev python3-dev

# Install Python dependencies
pip install torch numpy pybind11

# Clone Triton
git clone https://github.com/triton-lang/triton.git

# Build with plugin
cd triton
TRITON_PLUGIN_DIRS=/path/to/triton_tvm \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_WITH_CCACHE=true \
python3 -m pip install -e python --no-build-isolation -v
```

### Triggering Manual Builds

You can manually trigger workflows from the GitHub Actions UI:
1. Go to the "Actions" tab in your repository
2. Select the workflow you want to run
3. Click "Run workflow"
4. Fill in any required inputs

## Caching

The CI workflow uses caching to speed up builds:
- **pip cache**: Python packages
- **ccache**: C++ compilation cache

This significantly reduces build times for subsequent runs.

## Notes

- Tests requiring CUDA will be skipped in the CI environment but won't fail the build
- The release workflow automatically creates GitHub releases when tags are pushed
- All workflows use `continue-on-error: true` for tests to ensure builds complete even if tests fail
- Artifacts are retained for 7 days in CI builds

## Customization

To customize the workflows:
- Modify Python version in `setup-python` action
- Add/remove dependencies in installation steps
- Adjust cache keys and paths
- Change retention days for artifacts
- Add additional test suites or checks
