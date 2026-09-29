"""Sphinx configuration for the TIRx Harness documentation."""

project = "TIRx Harness"
author = "TIRx contributors"
extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
]
source_suffix = {".md": "markdown"}
root_doc = "index"
language = "en"
exclude_patterns = ["_build", "_hosting", "README.md", "requirements.*", ".DS_Store"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 5

# API definitions and operational procedures retain their canonical owners.
extlinks = {
    "repo": ("https://github.com/mlc-ai/TIRx-harness/blob/main/%s", "%s"),
    "kernels": ("https://github.com/mlc-ai/TIRx-kernels/blob/main/%s", "%s"),
    "kcoral": ("https://github.com/mlc-ai/kcoral/blob/main/%s", "%s"),
}

html_theme = "furo"
html_baseurl = "https://tirxharness.mlc.ai/docs/"
html_title = f"{project} documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
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
