# Contributing to Historical Fidelity Simulator

Thank you for your interest in contributing to the Historical Fidelity Simulator project! This document outlines the process and standards for contributing.

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style & Standards

- Follow PEP 8 style guide
- Use type hints for all function definitions
- Maximum line length: 88 characters (Black formatter)
- Run formatters before committing:
  ```bash
  black .
  mypy .
  pylint src/historical_fidelity_simulator
  ```

## Git Workflow

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/description
   ```
3. Make your changes following our standards
4. Write/update tests
5. Update documentation
6. Commit using conventional commits format:
   ```
   type(scope): description

   [optional body]

   [optional footer]
   ```
7. Push and create a pull request

## Pull Request Process

1. Ensure all tests pass
2. Update relevant documentation
3. Add/update example notebooks if needed
4. Request review from maintainers
5. Address review feedback

## Testing Requirements

- Write unit tests for new functionality
- Maintain test coverage above 90%
- Test both classical and quantum modes
- Include performance tests for core operations
- Verify energy conservation and state normalization

## Documentation

- Add docstrings for all public APIs
- Include example usage
- Reference equations/papers where relevant
- Update README.md if needed

## Performance Considerations

- Use vectorized operations where possible
- Implement parallel processing for heavy computations
- Cache expensive calculations
- Monitor and optimize memory usage

## Questions or Need Help?

Open an issue on GitHub or contact the maintainers. 