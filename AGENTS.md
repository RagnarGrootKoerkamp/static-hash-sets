# Benchmarking `static_hash_set`
- Keep the user-friendly commands in `justfile` up to date. It also contains some commonly used flags.
- Add LLM-reusable commands to a separate section of the `justfile`, with reusable arguments so that they can be invoked directly via `just XYZ`.
- by default, bench and profile on `hellman`; see `eval/latest-hellman.{csv,html}`
- pin the frequeny before server benches; `hellman` at 2.551GHz
- No need to run a full benchmark suite for quick perf testing
- SSH to `ae-new` using `~/.ssh/config`, and then use slurm to run on `hellman` (AMD EPYC) and optionally `floyd` (Intel XEON)
- do not add unit tests. Instead, rely on a single end-to-end fuzz-test shared between all hashset implementations.
