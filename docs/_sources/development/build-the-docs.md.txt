# Build the docs

[Sphinx](https://www.sphinx-doc.org/) builds the website,
[MyST](https://myst-parser.readthedocs.io/) reads Markdown, and
[Furo](https://pradyunsg.me/furo/) supplies the theme. The documentation has its
own locked Python dependencies and builds without installing the harness,
initializing submodules, or setting up a GPU.
The build does not execute kernel or remote-server examples.

## Build and preview

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root,
set `DOC_ENV` and `DOC_OUTPUT` to locations for this checkout. When using an
isolated task directory, keep both locations inside that directory.

```bash
DOC_REPO="$PWD"
DOC_ENV="$DOC_REPO/.local/docs-venv"
DOC_OUTPUT="$DOC_REPO/docs/_build/html"
uv venv --python 3.12 "$DOC_ENV"
uv pip install --python "$DOC_ENV/bin/python" -r "$DOC_REPO/docs/requirements.txt"
"$DOC_ENV/bin/python" -m sphinx -E -a -b html -n -W --keep-going \
  "$DOC_REPO/docs" "$DOC_OUTPUT"
"$DOC_ENV/bin/python" -m http.server 8018 --bind 127.0.0.1 \
  --directory "$DOC_OUTPUT"
```

Open <http://127.0.0.1:8018>. Stop the foreground server with Ctrl+C. Rebuild
after editing a page, then reload the browser. There is no editable package
installation: the build reads the Markdown files directly.

`-E -a` rebuilds all pages without reusing the saved document environment.
`-n` checks references, `-W` fails on warnings, and `--keep-going` reports as
many issues as possible. After moving or deleting pages, remove the generated
output directory before rebuilding so stale pages are not left in the site.
The theme, fonts, search, styles, and scripts are served locally.

### Preview from a remote workspace

Keep the server running on the machine containing the checkout. In a terminal
on your own computer, forward its port using your configured SSH host
alias:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 8018:127.0.0.1:8018 YOUR_SSH_HOST
```

Then open <http://127.0.0.1:8018> on your computer. If the local port is already
in use, change the first `8018` in the forwarding argument and the browser URL.
If the remote port is occupied, choose another port for both the server and
the final `8018` in the forwarding argument. A server inside an isolated
container must be reachable from the SSH host for forwarding to work.

## Maintain the documentation

- `index.md` owns the overview and the Get Started, Components, and Development
  navigation groups.
- `installation.md` owns prerequisites, package installation, and skill installation.
- `quick-start.md` introduces using the skills with an agent on a concrete kernel.
- `optimization-runs.md` covers optimization runs under Get Started. Its diagram is
  maintained in `_static/agent-loop.svg` and included directly in the page.
- `components/` introduces kernel authoring, analysis, and remote execution.
- `development/` covers workload registration, contributions, fixes, and this guide.
- `_static/custom.css` extends Furo's color variables for cards and the diagram.
  Check light and dark modes, narrow screens, and keyboard navigation after
  changing the styles. The theme supplies search and mobile navigation.

Add every new page to a `toctree` in `index.md`. Use relative Markdown links
between pages so Sphinx checks their targets. Preserve heading anchors when
other pages link to them.

Keep operational procedures in their existing skill references and link to
them using the `repo` role configured in `conf.py`. Full external API
definitions remain in their owning repositories.
Source links to `main` show current source; readers must match interfaces to
their installed package revision. Dependency lists and task declarations
remain authoritative for package versions and workload contracts.

Update `docs/requirements.in`, then regenerate the dependency lock file:

```bash
uv pip compile --python-version 3.12 docs/requirements.in -o docs/requirements.txt
```

Reinstall the requirements and run the strict build after updating dependencies.

## Automated checks

The [Documentation workflow](https://github.com/mlc-ai/TIRx-harness/blob/main/.github/workflows/docs.yml)
runs on pull requests, pushes to `main`, and manual dispatch. It installs only
the locked documentation dependencies with Python 3.12, then runs the same
strict HTML build as the local command. These CI checks need no kernel
dependencies, submodules, or GPU.

A successful build uploads a `documentation-html` artifact containing the
complete website, with the documentation under `docs/`. To preview it, download
and extract the artifact from the workflow run, then serve the extracted directory:

```bash
python -m http.server 8018 --bind 127.0.0.1 --directory /path/to/extracted/artifact
```

Open `http://127.0.0.1:8018/docs/`. The artifact root redirects to that path.

## Check external links

External link checking is separate from the HTML build because many source
links require repository access. With the same documentation environment:

```bash
"$DOC_ENV/bin/python" -m sphinx -b linkcheck -W --keep-going \
  "$DOC_REPO/docs" "$DOC_REPO/docs/_build/linkcheck"
```

This needs a network connection. Sphinx does not inherit GitHub CLI login
credentials, so private repository links can return 404 even when the files
exist. Check those targets in an authenticated GitHub session. Local preview
URLs and example server addresses are excluded from this optional check.
