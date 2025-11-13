# Contributing to DetectSleepStates

Thank you for your interest in contributing to DetectSleepStates! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Guidelines](#development-guidelines)
- [Submission Guidelines](#submission-guidelines)
- [Community](#community)

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or experience level.

### Our Standards

**Positive behaviors include:**
- Being respectful and inclusive
- Accepting constructive feedback
- Focusing on what's best for the project
- Showing empathy towards others

**Unacceptable behaviors include:**
- Harassment or discriminatory language
- Personal attacks or trolling
- Sharing others' private information
- Other unprofessional conduct

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

1. **Python 3.7+** installed
2. **Git** for version control
3. **Jupyter Notebook** for development
4. Basic understanding of machine learning and data science

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DetectSleepStates.git
   cd DetectSleepStates
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/zdangz/DetectSleepStates.git
   ```

### Set Up Development Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn jupyter pyarrow
   ```

3. Start Jupyter:
   ```bash
   jupyter notebook
   ```

## 💡 How to Contribute

### Types of Contributions

We welcome various types of contributions:

#### 1. Bug Reports

Found a bug? Please open an issue with:
- Clear, descriptive title
- Steps to reproduce the problem
- Expected vs actual behavior
- System information (OS, Python version, etc.)
- Code snippets or error messages

**Example Bug Report**:
```markdown
**Title**: Memory error when processing large datasets

**Description**: 
When running the notebook with the full train_series.parquet file,
I encounter a MemoryError at the merging step.

**Steps to Reproduce**:
1. Load full train_series.parquet (127M rows)
2. Run merge operation in Section 3
3. Error occurs

**Environment**:
- OS: Ubuntu 20.04
- Python: 3.8.10
- RAM: 8GB

**Error Message**:
```
MemoryError: Unable to allocate 2.1 GiB for array
```

#### 2. Feature Requests

Have an idea for improvement? Open an issue with:
- Clear description of the feature
- Use case and benefits
- Possible implementation approach
- Any relevant examples or references

**Example Feature Request**:
```markdown
**Title**: Add LSTM model for sequence prediction

**Description**:
Implement an LSTM-based deep learning model to capture
temporal dependencies in the accelerometer data.

**Benefits**:
- Better capture of sequential patterns
- Potentially improved prediction accuracy
- Provides baseline for comparing with Random Forest

**Proposed Implementation**:
- Create new notebook: `lstm_model.ipynb`
- Use TensorFlow/Keras for implementation
- Include hyperparameter tuning
```

#### 3. Code Contributions

Want to submit code? Great! Follow these steps:

1. **Check existing issues** to avoid duplicate work
2. **Create an issue** describing your planned changes
3. **Wait for feedback** before starting major work
4. **Follow the development guidelines** below
5. **Submit a pull request** when ready

#### 4. Documentation Improvements

Documentation is crucial! You can help by:
- Fixing typos or unclear explanations
- Adding code examples
- Improving existing guides
- Translating documentation

#### 5. Testing and Validation

Help ensure code quality by:
- Testing features on different systems
- Validating results and findings
- Reporting edge cases
- Suggesting test cases

## 🛠️ Development Guidelines

### Code Style

#### Python Code

Follow [PEP 8](https://pep8.org/) style guide:

```python
# Good
def calculate_rolling_average(data, window_size=10):
    """Calculate rolling average for time series data.
    
    Args:
        data (pd.Series): Input time series
        window_size (int): Size of rolling window
        
    Returns:
        pd.Series: Rolling average values
    """
    return data.rolling(window=window_size, min_periods=1).mean()

# Avoid
def calc(d,w=10):
    return d.rolling(window=w,min_periods=1).mean()
```

**Key Points**:
- Use descriptive variable names
- Add docstrings to functions
- Use 4 spaces for indentation
- Keep lines under 100 characters
- Add comments for complex logic

#### Jupyter Notebooks

**Structure**:
- Use markdown cells for section headers
- One logical operation per code cell
- Clear outputs before committing
- Include cell execution numbers

**Example**:
```markdown
## 1. Data Loading

Load the training data and display basic statistics.
```

```python
# Load training data
train_series = pd.read_parquet(path1)
print(f"Loaded {len(train_series):,} rows")
print(train_series.head())
```

### Commit Messages

Write clear, descriptive commit messages:

**Format**:
```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic changes)
- `refactor`: Code restructuring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:

```bash
# Good
git commit -m "feat: Add LSTM model implementation

- Implement LSTM architecture for sequence prediction
- Add training and evaluation functions
- Update documentation with usage examples

Closes #42"

# Good (simple change)
git commit -m "fix: Handle missing values in rolling window calculation"

# Avoid
git commit -m "updates"
git commit -m "fix stuff"
```

### Branch Naming

Use descriptive branch names:

**Format**: `<type>/<short-description>`

**Examples**:
```bash
# Feature branch
git checkout -b feat/lstm-model

# Bug fix branch
git checkout -b fix/memory-error

# Documentation branch
git checkout -b docs/update-readme
```

### Testing

Before submitting changes:

1. **Test your changes**:
   ```python
   # Verify functionality
   assert result.shape == expected_shape
   assert result.isna().sum() == 0
   ```

2. **Check for errors**:
   - Run all notebook cells sequentially
   - Verify outputs are correct
   - Check for warnings or errors

3. **Validate on different data**:
   - Test with small sample
   - Test with edge cases
   - Verify memory usage

### Documentation

Update documentation for any changes:

1. **Code Comments**: Explain complex logic
2. **Docstrings**: Document functions and classes
3. **README**: Update if adding new features
4. **DEVELOPER_GUIDE**: Add technical details
5. **CONTRIBUTING**: Update guidelines if needed

## 📤 Submission Guidelines

### Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feat/your-feature
   ```

3. **Make your changes**:
   - Write code
   - Add documentation
   - Test thoroughly

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: Your descriptive message"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feat/your-feature
   ```

6. **Open a Pull Request**:
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Changes Made
- List key changes
- Include implementation details
- Note any breaking changes

## Testing
- [ ] Tested on local machine
- [ ] Verified with sample data
- [ ] Checked for memory usage
- [ ] All notebook cells run successfully

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] No new warnings or errors
- [ ] Commit messages are clear and descriptive

## Related Issues
Closes #issue_number
```

### Review Process

After submitting a PR:

1. **Automated checks** will run (if configured)
2. **Maintainers will review** your code
3. **Feedback may be provided** for improvements
4. **Address review comments** and push updates
5. **PR will be merged** once approved

**Response Time**:
- Initial review: 3-7 days
- Follow-up reviews: 1-3 days
- Be patient and responsive to feedback

## 🎯 Contribution Ideas

### Beginner-Friendly

Good first contributions:

1. **Documentation**:
   - Fix typos
   - Improve explanations
   - Add code examples

2. **Code Cleanup**:
   - Add comments
   - Improve variable names
   - Format code consistently

3. **Testing**:
   - Test on different systems
   - Validate edge cases
   - Report issues

### Intermediate

For more experienced contributors:

1. **Feature Engineering**:
   - Add new features
   - Implement feature selection
   - Optimize calculations

2. **Model Improvements**:
   - Try different algorithms
   - Tune hyperparameters
   - Implement cross-validation

3. **Performance**:
   - Optimize memory usage
   - Speed up computations
   - Add parallel processing

### Advanced

For experienced ML practitioners:

1. **Deep Learning**:
   - Implement LSTM/GRU models
   - Add attention mechanisms
   - Create ensemble models

2. **Advanced Features**:
   - FFT-based features
   - Wavelet transforms
   - Domain-specific features

3. **Production**:
   - Create API endpoint
   - Add model monitoring
   - Implement MLOps pipeline

## 🌟 Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes
- Credited in documentation

Thank you for contributing to DetectSleepStates!

## 📞 Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Email**: Contact maintainers for sensitive issues

### Communication Guidelines

- Be respectful and professional
- Stay on topic
- Provide context and details
- Be patient with responses
- Help others when you can

## 📚 Additional Resources

### Learning Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)
- [Kaggle Learn](https://www.kaggle.com/learn)

### Project Resources

- [README.md](README.md) - Project overview
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Technical details
- [Competition Page](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states)

## 📄 License

By contributing, you agree that your contributions will be subject to the same license terms as the project (see competition rules for data usage).

---

**Questions?** Open an issue or reach out to the maintainers!

**Happy Contributing! 🎉**
