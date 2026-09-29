# TIRx Harness documentation website

Generated documentation for [TIRx Harness](https://github.com/mlc-ai/TIRx-harness),
served at <https://tirxharness.mlc.ai/docs/>.

## Local preview

After extracting `artifact.tar` from the `github-pages` workflow artifact,
serve the extracted website's root:

```bash
python -m http.server 8018 --bind 127.0.0.1 --directory .
```

Open <http://127.0.0.1:8018/docs/>.
