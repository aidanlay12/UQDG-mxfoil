# Contributing to UQDG-mxfoil

Thank you for your interest in contributing to UQDG-mxfoil! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs or suggest features
- Provide clear descriptions and include relevant details
- Include error messages, operating system, and software versions

### Code Contributions

#### Getting Started
1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature/fix
4. Make your changes
5. Test your changes thoroughly
6. Submit a pull request

#### Development Setup

**Python Environment:**
```bash
cd UQDG_Python/
pip install -e .
# For GCI analysis:
git clone https://github.com/ORNL/cfd-verify.git
cd cfd-verify
pip install -e .
```

**MATLAB Environment:**
```matlab
addpath('UQDG_MATLAB/src');
```

#### Code Style
- **Python**: Follow PEP 8 style guidelines
- **MATLAB**: Use consistent naming conventions and documentation
- Add comments for complex algorithms
- Include docstrings for all functions

#### Testing
- Test your changes with the provided tutorials
- Ensure backward compatibility
- Verify that examples still work
- Test on different operating systems if possible

### Documentation
- Update README files if adding new features
- Add inline comments for complex code
- Update tutorials if functionality changes
- Keep documentation clear and concise

### Pull Request Process
1. Ensure your code follows the project's style guidelines
2. Update documentation as needed
3. Add or update tests for new functionality
4. Describe your changes clearly in the pull request
5. Reference any related issues

## Code of Conduct
- Be respectful and professional in all interactions
- Focus on constructive feedback
- Help maintain a welcoming environment for all contributors

## Questions?
If you have questions about contributing, feel free to:
- Open an issue for discussion
- Contact the maintainer: alay12@vols.utk.edu

Thank you for helping improve UQDG-mxfoil!
