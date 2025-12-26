# triton-tvm

[![CI](https://github.com/douyixuan/triton-tvm/actions/workflows/ci.yml/badge.svg)](https://github.com/douyixuan/triton-tvm/actions/workflows/ci.yml)
[![Code Quality](https://github.com/douyixuan/triton-tvm/actions/workflows/lint.yml/badge.svg)](https://github.com/douyixuan/triton-tvm/actions/workflows/lint.yml)

A transpiler from Triton to TVM TensorIR.

## Features

- Accepts `ttgpu` dialect and reuses existing optimizations as a baseline.
- End-to-end integration.

## CI/CD

This project uses GitHub Actions for continuous integration and release automation:

- **CI Workflow**: Automatically builds and tests on every push and pull request
- **Release Workflow**: Creates versioned releases when tags are pushed (e.g., `v1.0.0`)
- **Code Quality**: Runs linting and formatting checks

See [.github/workflows/README.md](.github/workflows/README.md) for detailed documentation.

## Usage

Similar to [triton-shared](https://github.com/microsoft/triton-shared). See their documentation for more details.

To build this repo clone `triton-tvm` to a folder called `triton_tvm` (notice the **underscore**). You need to set the `TRITON_PLUGIN_DIRS` environment variable to the location of your `triton-tvm` directory for `triton` to find it.

After installation, just add the following lines to your Triton script:

```python
from triton.backends.triton_tvm.driver import TVMDriver
triton.runtime.driver.set_active(TVMDriver())
```

See [the examples](example) for runnable tests.

## Design

### TVM Dialect

An MLIR dialect for TVM TensorIR is devised.

| TensorIR | TVM Dialect | Notes |
|----------|-------------|-------|
| `T.var()` | `tvm.var` | |
| `T.match_buffer` | `tvm.match_buffer` | |
| `T.alloc_buffer` | `tvm.alloc_buffer` | |
| `for k in range(n): ...` | `scf.for {tvm.for_kind = serial}` | `serial`, `parallel`, `vectorized`, `unroll`, `thread_binding` |
| `T.block` | `tvm.block` | |
| `T.where` | `tvm.where` | |
| `T.axis.spatial(k)` | `tvm.axis spatial %k` | `spatial`, `reduce`, `scan`, `opaque` |
| `A[i, j]` | `tvm.ref %A[%i, %j]` | |
| `T.if_then_else` | `tvm.if_then_else` | |
| `T.reads` | `tvm.read` | |
| `T.writes` | `tvm.write` | |
| `A[i, j] = ...` | `tvm.assign %ref = %value` | |
| `T.init` | `tvm.init` | |
| `T.min_value` | `tvm.min_value` | |
| `T.max_value` | `tvm.max_value` | |

### Conversion

Basically, by pattern matching.

| Triton Dialect | TVM Dialect |
|--------|-----|
| `tt.load` | `tvm.block` + `tvm.if_then_else` |
| `tt.store` | `tvm.block` + `tvm.where` |
| `tt.reduce` | Materialize with `tvm.block`, using implicit reduction axes. |
| `arith.*` | Do NOT materialze. Keep in the form. |
| `for k: ...` | `scf.for {tvm.for_kind = serial}` |
| `for k: ptrs += stride` | `ptrs = offset + k * stride` |
| `for k: acc += ...` | Materialize with `tvm.block`, with `tvm.axis reduce k` |

## Limitations

No support for:

- `tt.make_tensor_ptr` and `tt.advance`
- `tt.dot`
- Dynamic shapes. You have to mark any shape-related parameters as `tl.constexpr`.
