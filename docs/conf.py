"""Sphinx configuration for the TIRx Harness documentation."""

from pathlib import Path

from sphinx_design.icons import get_octicon

project = "TIRx Harness"
author = "TIRx contributors"
extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "autoapi.extension",
]
source_suffix = {".md": "markdown"}
root_doc = "index"
language = "en"
exclude_patterns = ["_build", "_hosting", "README.md", "requirements.*", ".DS_Store"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 5

# Read source without importing the harness or its native dependencies. The
# reference pages select user-facing objects instead of publishing every module.
autoapi_dirs = [str(Path(__file__).resolve().parents[1] / "tools/src/tirx_harness")]
autoapi_generate_api_docs = False
autoapi_add_toctree_entry = False
autosummary_generate = False
autodoc_member_order = "bysource"
python_use_unqualified_type_names = True
# External types and the finding record described on the checker page do not
# have local reference pages. Keep all other missing-reference checks enabled.
nitpick_ignore = [
    ("py:class", "collections.abc.Callable"),
    ("py:class", "collections.abc.Iterable"),
    ("py:class", "collections.abc.Mapping"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "pathlib.Path"),
    ("py:class", "tvm.tir.PrimFunc"),
    ("py:class", "Finding"),
]


def resolve_public_api_reference(app, env, node, contnode):
    """Link implementation annotations to their documented public re-exports."""
    if node.get("refdomain") != "py":
        return None
    for name, obj in env.autoapi_all_objects.items():
        if obj.obj.get("original_path") == node["reftarget"]:
            resolved = env.domains["py"].resolve_xref(
                env, node["refdoc"], app.builder, node["reftype"], name, node, contnode
            )
            if resolved is not None:
                return resolved
    return None


def setup(app):
    app.connect("missing-reference", resolve_public_api_reference)


# API definitions and operational procedures retain their canonical owners.
extlinks = {
    "repo": ("https://github.com/mlc-ai/TIRx-harness/blob/main/%s", "%s"),
    "kernels": ("https://github.com/mlc-ai/TIRx-kernels/blob/main/%s", "%s"),
    "kcoral": ("https://github.com/mlc-ai/kcoral/blob/main/%s", "%s"),
}

html_theme = "furo"
templates_path = ["_templates"]
html_baseurl = "https://tirxharness.mlc.ai/docs/"
html_title = f"{project} documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_context = {"github_icon": get_octicon("mark-github", height="1.25rem")}
html_theme_options = {
    "source_repository": "https://github.com/mlc-ai/TIRx-harness/",
    "source_branch": "main",
    "source_directory": "docs/",
}
html_show_sourcelink = True
html_copy_source = True
html_show_copyright = False
html_use_index = False
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
linkcheck_ignore = [r"http://(?:localhost|127\.0\.0\.1|your-server)(?::\d+)?(?:/.*)?$"]
