- Author every production GPU kernel in TIRx-lite:
  `import tirx_kernels.tirx_lite as txl`. Every kernel launched by the timed callable
  must originate from `@txl.kernel`.
- TIRx-lite is the source language; TIRx is its generated IR and compiler pipeline.
  Do not author kernel entries through raw TVMScript/TIRx parser or builder
  APIs, or through another GPU language or runtime.
- Keep TIRx-lite's construction-time low-level IR validation enabled. Passing
  `check_ir=False` invalidates the candidate. Do not grant `allowed_func_calls`
  unless the task contract explicitly names the permitted runtime call.
