# Contributing to CS-SMol-AndroR-model-publication

We welcome contributions to improve the androgen receptor prediction models and analysis tools! This guide outlines how to contribute effectively.

## Types of Contributions

### Bug Reports and Feature Requests
- Report bugs via GitHub Issues
- Suggest new features or improvements
- Propose better evaluation metrics
- Request documentation improvements

### Code Contributions
- Bug fixes
- Performance improvements
- New analysis tools
- Additional model implementations
- Enhanced visualizations

### Documentation Contributions
- Improve README or methodology documentation
- Add examples and tutorials
- Translate documentation
- Fix typos or unclear explanations

## Development Workflow

### 1. Setup Development Environment

```bash
# Fork the repository on GitHub
git clone https://github.com/your-username/CS-SMol-AndroR-model-publication.git
cd CS-SMol-AndroR-model-publication

# Create virtual environment
python -m venv dev_env
source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate

# Install dependencies + development tools
pip install -r requirements.txt
pip install black flake8 pytest jupyter
```

### 2. Make Changes

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Test your changes
python -m pytest  # if tests exist
python -c "import ml, analysis, utils"  # basic import test
```

### 3. Code Quality

```bash
# Format code
black *.py

# Check linting
flake8 *.py --max-line-length=88
```

### 4. Submit Changes

```bash
# Commit changes
git add .
git commit -m "Add feature: your clear description"

# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

## Code Standards

### Python Style
- Follow PEP 8 style guide
- Use Black formatter (line length: 88)
- Include docstrings for functions and classes
- Use type hints where helpful

### Example Function Documentation
```python
def compute_sample_weights(y_train_fold: np.ndarray) -> np.ndarray:
    """
    Compute balanced sample weights for training data.
    
    Parameters
    ----------
    y_train_fold : np.ndarray
        Training labels for current fold
        
    Returns
    -------
    np.ndarray
        Sample weights for balanced training
        
    Examples
    --------
    >>> y = np.array(['active', 'inactive', 'active'])
    >>> weights = compute_sample_weights(y)
    >>> len(weights) == len(y)
    True
    """
```

### Notebook Guidelines
- Clear cell organization with markdown headers
- Remove output before committing (except for demo notebooks)
- Include brief explanations of analysis steps
- Use descriptive variable names

## Testing

### Current Testing Status
This repository currently has minimal formal testing. Contributions to improve testing are welcome!

### Suggested Tests
- Unit tests for utility functions
- Integration tests for ML pipeline
- Validation tests for molecular fingerprint generation
- Example data for testing

### Running Tests
```bash
# If tests exist
python -m pytest tests/

# Manual verification
python test_imports.py
```

## Documentation

### Adding Examples
```python
# Add usage examples to function docstrings
# Create demo scripts in examples/ folder
# Update README with new features
```

### Updating Documentation
- Update relevant .md files for new features
- Keep methodology documentation current
- Add references for new algorithms

## Specific Areas for Contribution

### High Priority
1. **Testing Framework**: Add comprehensive tests
2. **Example Data**: Create small demo datasets
3. **Model Persistence**: Better model saving/loading
4. **Performance Optimization**: Speed improvements
5. **Additional Metrics**: More evaluation measures

### Medium Priority
1. **Alternative Algorithms**: Random Forest, SVM implementations
2. **Feature Engineering**: Additional molecular descriptors
3. **Visualization**: Enhanced plotting functions
4. **Cross-validation**: Additional CV strategies
5. **Documentation**: More detailed examples

### Research Extensions
1. **Deep Learning**: Graph neural network implementations
2. **Multi-task Learning**: Simultaneous endpoint prediction
3. **Uncertainty Quantification**: Conformal prediction
4. **External Validation**: Integration with public datasets
5. **Chemical Interpretation**: Advanced SHAP analysis

## Review Process

### Pull Request Guidelines
1. **Clear Description**: Explain what and why
2. **Small Changes**: Keep PRs focused and manageable
3. **Tests**: Include tests for new functionality
4. **Documentation**: Update docs for new features
5. **Backwards Compatibility**: Avoid breaking existing code

### Review Criteria
- Code quality and style
- Scientific validity
- Documentation completeness
- Test coverage
- Performance impact

## Communication

### Getting Help
- **GitHub Issues**: For questions and discussions
- **Pull Request Comments**: For code-specific feedback
- **Documentation**: Check existing docs first

### Reporting Issues
Please include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version)
- Error messages or screenshots

## Scientific Contributions

### Model Improvements
- Validate changes on independent data
- Compare performance to baseline
- Document methodology clearly
- Consider regulatory implications

### Data Contributions
- Ensure data quality and provenance
- Follow FAIR data principles
- Include proper citations
- Respect licensing restrictions

## Recognition

Contributors will be acknowledged in:
- GitHub contributor list
- Future publications (with permission)
- CONTRIBUTORS.md file
- Release notes for significant contributions

## Code of Conduct

We follow the Contributor Covenant Code of Conduct:
- Be respectful and inclusive
- Focus on constructive feedback
- Welcome newcomers and learning
- Maintain professional interactions

## Questions?

Feel free to:
- Open a GitHub Issue for questions
- Suggest improvements to this guide
- Propose new contribution areas

Thank you for contributing to computational toxicology research!