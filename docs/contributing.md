# Contributing

Contributions are welcome! This document outlines how to set up a development environment and submit changes.

---

## Development Setup

```bash
git clone https://github.com/clebiomojunior/data-science-toolkit.git
cd data-science-toolkit
pip install -e .[dev]
```

---

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
- Write [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) for all public functions and classes.
- Use type hints where applicable.

---

## Testing

Run tests with pytest:

```bash
pytest
```

With coverage:

```bash
pytest --cov=dstoolkit --cov-report=term-missing
```

The CI pipeline enforces a minimum coverage of 80%.

---

## Documentation

Build the docs locally:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.

---

## Pull Request Process

1. Create a feature branch from `main`.
2. Add tests for new functionality.
3. Ensure all tests pass and coverage does not decrease.
4. Update documentation if the public API changes.
5. Submit a pull request with a clear description of the changes.

---

## Reporting Issues

Report bugs and feature requests on the [issue tracker](https://github.com/clebiomojunior/data-science-toolkit/issues).
