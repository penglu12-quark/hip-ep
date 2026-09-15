<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Custom-kernel test pipeline

One command to build and run every kernel suite under
`lib/Runtime/Kernels/test/example` on a given architecture, with a single
pass/fail summary at the end.

---

## TL;DR

```bat
scripts\run_full_pipeline_kernel_tests.bat --arch gfx1151
```

Exits non-zero if any suite fails, so it drops straight into CI or a pre-push
hook.

---

## What it runs

Each suite is driven through **its own `Makefile`** rather than by re-issuing
`hipcc` commands from the pipeline. That matters: the recipes are not uniform
and they drift. `gemm_fp16u2` and `gemm_fp16i8` link an extra
`autotune_stub.cpp` (the kernel references
`hipdnn_ep::matmul_nbits_autotune::resolve()`, whose real definition pulls in
flatbuffers and a CMake-generated header), while `gemm_fp16u3` passes the
autotune include path to the test TU instead and links no stub. Duplicating
that here would rot silently the next time a suite changes.

| Suite | Directory | Driven as |
|---|---|---|
| MatMulNBits u2 | `gemm_fp16u2` | `make test SIZE=... GS=...` |
| MatMulNBits u3 | `gemm_fp16u3` | `make test SIZE=... GS=...` |
| MatMulNBits i8 | `gemm_fp16i8` | `make test SIZE=... GS=...` |
| GQA autotune sweep | `gqa/autotune` | `make run` |

The GQA entry is a shape sweep rather than a pass/fail test: it takes minutes,
writes CSV and log output into its own directory, and pipes through `tee`
(so it needs MSYS2/WSL coreutils, not just `make`). `--quick` skips it.

The MatMulNBits suites run on shapes chosen to straddle the dispatch boundary
rather than to be exhaustive: `1x2880x5120` is a decode shape (M=1) that takes
the GEMV path, `128x2880x5120` is a small prefill that takes WMMA, and
`1x4096x2880` adds a second decode aspect ratio. `--quick` keeps only the
first. Each suite verifies against a NumPy reference before reporting GFLOPS
and GB/s; the pipeline reports whatever the suite reports and imposes no
tolerance of its own.

### Suites that are skipped

`gemm_fp16u4`, `gqa/decode` and `gqa/prefill` have test drivers but no
`Makefile`, so they cannot be built from a clean checkout and the pipeline
reports them as `SKIP` with that reason. This is a side effect of
[`.gitignore`](../.gitignore), whose line 7 is a bare rule that matches at
every level of the tree:

```
Makefile
```

The suites that *do* have one (`gemm_fp16u2`, `gemm_fp16u3`, `gemm_fp16i8`,
`gqa/autotune`) were force-added past that rule. Adding a negation such as
`!lib/Runtime/Kernels/test/example/**/Makefile` would let the remaining suites
be committed and picked up here automatically — the pipeline needs no change,
it already probes for the file.

---

## Prerequisites

- A **HIP SDK** (`hipcc`). The Makefiles default to `C:\AMD\Rocm\7.1` on
  Windows and `/mnt/c/AMD/ROCm/7.1` under WSL.
- **`make`** on `PATH` — from MSYS2, Git-for-Windows, or WSL. The Makefiles
  are the build recipe, so this is not an extra dependency the pipeline
  invents; it is what the per-suite READMEs already assume.
- **Python** with NumPy, for the reference-data generators.

---

## Usage

| Option | Default | Meaning |
|---|---|---|
| `--arch <gfxNNNN>` | `gfx1150` | offload architecture |
| `--hip-sdk <path>` | per-Makefile | HIP SDK root; validated before anything runs |
| `--python <exe>` | per-Makefile | interpreter for the data generators |
| `--make <path>` | found on `PATH` | `make` executable |
| `--suite <name>` | `all` | `all`, `matmul`, or `gqa` |
| `--gs <n>` | `128` | MatMulNBits group size |
| `--quick` | off | decode shape only, one per suite |

`--arch` is always forwarded, which normalizes an inconsistency: the
MatMulNBits Makefiles default to `gfx1150` and `gqa/autotune` to `gfx1151`. The
pipeline runs every suite on one architecture, defaulting to `gfx1150`; pass
`--arch` to pick your own. `--hip-sdk` and
`--python` are left unset unless you pass them, so each Makefile keeps its own
OS-aware default rather than having a Windows path forced on a WSL build.

```bat
REM everything, on a Strix Halo part
scripts\run_full_pipeline_kernel_tests.bat --arch gfx1151

REM just the quantized GEMMs, fast smoke run before pushing
scripts\run_full_pipeline_kernel_tests.bat --suite matmul --quick

REM non-default SDK location and group size
scripts\run_full_pipeline_kernel_tests.bat --hip-sdk D:\rocm --gs 32
```

An arch that does not match your card shows up as a runtime failure, not a
build error.

---

## Output

```
============================================================
 custom_kernels test pipeline
============================================================
  make      : C:\msys64\usr\bin\make.exe
  arch      : gfx1151
  suite     : all
  group size: 128

------------------------------------------------------------
 MatMulNBits u2
------------------------------------------------------------
  [make] test SIZE=1x2880x5120 GS=128
  ...

============================================================
 Summary  (arch gfx1151)
============================================================
  PASS  matmul u2 1x2880x5120
  PASS  matmul u2 128x2880x5120
  ...
  SKIP  matmul u4 [gemm_fp16u4 has no Makefile]
  PASS  gqa autotune
  SKIP  gqa decode [gqa\decode has no Makefile]
------------------------------------------------------------
  passed: 10    failed: 0    skipped: 3
============================================================
```

A failing suite prints the step it died at and the run continues, so one
broken kernel does not hide the state of the others. Exit status is `1` if
anything failed, `2` for a usage error, `0` otherwise. Skips do not fail the
run.

---

## Caveats

**Windows only.** The example Makefiles detect WSL and shell out to `cmd.exe`,
but this entry point is batch. Under WSL, invoke the per-suite `make` targets
directly. A Python rewrite would be the right fix if these need to run in
Linux CI.

**Scope.** This covers the hand-written HIP kernels in
`lib/Runtime/Kernels/hip` only — not the MLIR lowering, the ONNX frontend, or
the runtime wrappers in `lib/Runtime/real`.

---

## See also

- [`quick_start.md`](quick_start.md) — building the EP itself
- [`supported-operations.md`](supported-operations.md) — operator coverage
- `lib/Runtime/Kernels/test/example/*/README.md` — per-suite detail, including
  the quantization format conventions each driver assumes
