from pathlib import Path
import re
import shutil
from urllib.parse import urljoin

DOCS_INDEX = "docs/index.md"
REPO_URL = "https://github.com/Safenai/kcai-data-sampling"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _copy_if_changed(src: Path, dst: Path):
    """Copy file only if destination doesn't exist or content differs.

    Args:
        src: Source file path.
        dst: Destination file path.
    """
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        return
    shutil.copy(src, dst)


def _write_if_changed(dst: Path, content: str):
    """Write text content to file only if it differs from existing content.

    Args:
        dst: Destination file path.
        content: Text content to write.
    """
    if dst.exists() and dst.read_bytes() == content.encode():
        return
    dst.write_text(content)


def _rebase_docs_links(content: str) -> str:
    """Rebase ``docs/``-prefixed links so they resolve from inside the site.

    A document at the repo root may link to ``docs/quickstart.md`` — correct on
    GitHub and GitLab, where the reader stays at the root. Once the document is
    copied into ``docs/`` that same link points one directory too deep, so strip
    the prefix and let it resolve against the site root instead.

    Args:
        content: Markdown of a document that lives at the repo root.

    Returns:
        The markdown with its ``docs/``-prefixed relative links rebased.
    """
    content = content.replace("./docs/", "./")
    return content.replace("(docs/", "(")


def _copy_root_doc(src: Path, dst: Path):
    """Copy a repo-root markdown document into ``docs/``, rebasing its links.

    Args:
        src: Source file at the repo root.
        dst: Destination path inside ``docs/``.
    """
    if not src.exists():
        return
    _write_if_changed(dst, _rebase_docs_links(src.read_text()))


# ---------------------------------------------------------------------------
# docs/index.md pipeline  (single pass — no ordering dependencies)
# ---------------------------------------------------------------------------


def _build_index():
    """Copy README.md to docs/index.md, rewrite links for mkdocs site."""
    content = Path("README.md").read_text()

    # Rewrite relative paths that point into docs/ for GitHub/GitLab
    content = _rebase_docs_links(content)
    content = content.replace('src="docs/static/', 'src="./static/')
    content = content.replace(
        "(packages/",
        f"({REPO_URL}/tree/main/packages/",
    )
    content = content.replace(
        "(examples/",
        f"({REPO_URL}/tree/main/examples/",
    )

    # Inject repository link before "Available on PyPI" if not already present
    repo_link = (
        "- **[Repository](https://github.com/Safenai/kcai-data-sampling)**"
    )
    if "## Available on PyPI" in content and repo_link not in content:
        content = content.replace(
            "## Available on PyPI",
            f"{repo_link}\n\n## Available on PyPI",
        )

    _write_if_changed(Path(DOCS_INDEX), content)


# ---------------------------------------------------------------------------
# docs/examples/ pipeline  (single pass — copy source dirs in)
# ---------------------------------------------------------------------------


def _sync_dir(src: Path, dst: Path):
    """Sync a single directory by copying files to a destination.

    Args:
        src: Source directory path.
        dst: Destination directory path.
    """
    if not src.exists():
        return
    dst.mkdir(exist_ok=True)
    for src_file in sorted(src.iterdir()):
        if src_file.is_dir():
            continue
        _copy_if_changed(src_file, dst / src_file.name)


def _build_examples():
    """Copy examples/ to docs/examples/ (markdown is rendered in place)."""
    src_root = Path("examples")
    dst_root = Path("docs/examples")
    dst_root.mkdir(exist_ok=True)

    _sync_dir(src_root / "notebooks", dst_root / "notebooks")
    _sync_dir(src_root / "adapters", dst_root / "adapters")
    _sync_dir(src_root / "config", dst_root / "config")


# ---------------------------------------------------------------------------
# Simple file copies  (source root → docs/)
# ---------------------------------------------------------------------------


def copy_changelog():
    """Copy CHANGELOG.md from repo root to docs/."""
    _copy_root_doc(Path("CHANGELOG.md"), Path("docs/CHANGELOG.md"))


def copy_release_notes():
    """Copy RELEASE.md from repo root to docs/."""
    _copy_root_doc(Path("RELEASE.md"), Path("docs/RELEASE.md"))


def copy_package_readmes():
    """Copy package READMEs to docs/packages/."""
    packages_dir = Path("packages")
    docs_packages_dir = Path("docs/packages")
    docs_packages_dir.mkdir(exist_ok=True)

    for pkg in [
        "kcai-data-sampling-core",
        "kcai-data-sampling-images",
        "kcai-data-sampling-lama",
        "kcai-data-sampling-fgsm",
        "kcai-data-sampling-job",
        "kcai-data-sampling",
    ]:
        src = packages_dir / pkg / "README.md"
        if src.exists():
            _copy_if_changed(src, docs_packages_dir / f"{pkg}.md")


def _copy_coverage_report():
    """Copy coverage index.html to coverage_report.html for direct linking."""
    coverage_index = Path("docs/reports/htmlcov/index.html")
    if coverage_index.exists():
        _copy_if_changed(
            coverage_index, Path("docs/reports/htmlcov/coverage_report.html")
        )


# ---------------------------------------------------------------------------
# mkdocs page_markdown hook  (per-page, during render)
# ---------------------------------------------------------------------------


def _transform_examples_to_github(markdown: str, src_path: str) -> str:
    """Transform relative links to examples/ into GitHub URLs.

    Source docs/*.md files use relative paths like ``../examples/...`` that
    work locally and on GitHub/GitLab. On the mkdocs website these example
    files aren't served directly, so rewrite the links to permanent GitHub
    URLs.

    Args:
        markdown: Raw markdown content.
        src_path: Source path of the page relative to docs/.

    Returns:
        Markdown content with example links rewritten to GitHub URLs.
    """
    github_raw = f"{REPO_URL}/tree/main"

    if src_path.startswith("examples/"):
        page_dir = src_path.rsplit("/", 1)[0] + "/"

        def _repl(m):
            text, url = m.group(1), m.group(2)
            if url.startswith(("https://", "#", "/")):
                return m.group(0)
            if url.endswith(".md"):
                return m.group(0)
            if url.endswith((".py", ".yaml")):
                resolved = urljoin(page_dir, url)
                return f"[{text}]({github_raw}/{resolved})"
            return m.group(0)

        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _repl, markdown)

    # src_path is relative to docs/ directory.
    # E.g., "cli.md" → depth 0 → need "../" to reach repo root
    #        "configuration/features.md" → depth 1 → need "../../"
    depth = src_path.count("/")
    prefix = "../" * (depth + 1)
    return markdown.replace(f"({prefix}examples/", f"({github_raw}/examples/")


def _transform_package_links(markdown: str, src_path: str) -> str:
    """Transform relative package README links into docs/ page links.

    Source docs/*.md files use ``../packages/xxx/README.md`` paths that work
    on GitHub. On the mkdocs website these READMEs have been copied to
    ``docs/packages/xxx.md``, so rewrite the links accordingly.

    Args:
        markdown: Raw markdown content.
        src_path: Source path of the page relative to docs/.

    Returns:
        Markdown content with package links rewritten.
    """

    if src_path.startswith("examples/"):
        return markdown

    return re.sub(
        r"\(\.\./packages/([^/]+)/README\.md(#[^)]*)?\)",
        r"(packages/\1.md\2)",
        markdown,
    )


def page_markdown(markdown, page, config, files):  # NOSONAR
    """Transform relative links after page markdown is loaded.

    Args:
        markdown: Raw markdown content of the page.
        page: MkDocs page object.
        config: MkDocs configuration.
        files: Collection of all files in the docs directory.

    Returns:
        Transformed markdown content.
    """
    markdown = _transform_examples_to_github(markdown, page.file.src_path)
    markdown = _transform_package_links(markdown, page.file.src_path)
    return markdown


# ---------------------------------------------------------------------------
# mkdocs pre_build hook  (runs once before every build)
# ---------------------------------------------------------------------------


def pre_build(*args, **kwargs):
    """Pre-build hook: copy and transform files for mkdocs.

    All functions are self-contained — no implicit ordering dependencies.
    Each uses content-aware writes to avoid triggering unnecessary rebuilds.
    """
    copy_changelog()
    copy_release_notes()
    copy_package_readmes()
    _copy_coverage_report()
    _build_index()
    _build_examples()
