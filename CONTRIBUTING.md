# Contributing to taxonomic-rag-system

Thanks for your interest in contributing to the taxonomic-rag-system!

To submit PRs, please fill out the PR template along with the PR. If the PR
fixes an issue, don't forget to link the PR to the issue!

## Coding guidelines

For code style, we recommend the [PEP 8 style guide](https://peps.python.org/pep-0008/).

For docstrings we use [NumPy format](https://numpydoc.readthedocs.io/en/latest/format.html).

We use [Ruff](https://docs.astral.sh/ruff/) for code formatting and static code
analysis. Ruff checks various rules including [flake8](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8). The pre-commit hooks show errors which you need to fix before submitting a PR.

Last but not least, we use type hints in our code which is then checked using
[MyPy](https://mypy.readthedocs.io/en/stable/).

## Pre-commit hooks

Pre-commit hooks automatically run code quality checks before each commit. Once the Python virtual environment is set up and activated (see README.md), install and run pre-commit:

```bash
# Activate virtual environment first
source .venv/bin/activate

# Install pre-commit hooks (first time setup)
pre-commit install

# Run hooks on all files manually
pre-commit run --all-files

# Run hooks on staged files only
pre-commit run
```

**Note**: Alternatively, `uv` allows you to run pre-commit without explicitly activating the environment:

```bash
# Without activating the environment
uv run pre-commit install
uv run pre-commit run --all-files
```

### What the hooks check

Our pre-commit configuration includes:

- **Code formatting**: Ruff formatter for Python and Jupyter notebooks
- **Linting**: Ruff linter with extensive rule set
- **Type checking**: MyPy static type analysis
- **Security**: Private key detection
- **Quality**: Trailing whitespace, merge conflicts, YAML/TOML syntax
- **Documentation**: Doctest validation, typo detection
- **Dependencies**: UV lock file synchronization

The hooks will automatically fix many issues (like formatting) and flag others that require manual attention. All checks must pass before code can be committed.

## Git Workflow

This project follows [Gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) branching model:

- **`main`**: Production-ready code. All releases are tagged from this branch.
- **`develop`**: Integration branch for features. Default branch for development work.
- **Feature branches**: Created from `develop` for new features, merged back via PR.
- **Release branches**: Created from `develop` when preparing releases, merged to both `main` and `develop`.

### Branch Guidelines

- Create feature branches from `develop`: `git checkout -b feature/your-feature-name develop`
- Submit PRs to `develop` unless it's a hotfix for `main`
- Keep feature branches focused and short-lived
- Use descriptive branch names that indicate the feature or fix

## Claude Code Integration

This project includes configuration for [Claude Code](https://claude.ai/code), Anthropic's AI coding assistant, to improve development workflows and team collaboration.

### Configuration Files

- **`CLAUDE.md`**: Provides project context, architecture overview, and development guidance to Claude Code. This file is automatically loaded by Claude Code and should be kept current with project changes.

- **`.claude/settings.json`**: Shared team permissions for common development domains (LangChain, ChromaDB, Cohere, etc.) to streamline AI-assisted development without repeated permission prompts.

### Contributing to AI Configuration

- When adding new dependencies or changing architecture, update the relevant sections in `CLAUDE.md`
- If you frequently visit new domains during development, consider adding them to `.claude/settings.json`
- Test changes by using Claude Code and ensuring it has proper project context
- Keep the testing strategy and command examples current as the project evolves

### Why Version Control These Files?

These files are checked into git to ensure:

- **Consistent AI assistance** across all team members
- **Shared understanding** of project structure and conventions
- **Collaborative improvement** of AI tooling and project documentation
- **Onboarding efficiency** for new contributors (both human and AI)
