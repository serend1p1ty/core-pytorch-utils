# Developing Tips

## Code Checking

Before submitting a pull request, you should run `linter.sh`, which checks syntax and code style.

```
pip install -r requirements.txt
./linter.sh
```

## Release a New Version

We refer to the workflow introduced in [yapf](https://github.com/google/yapf/blob/main/HACKING.rst).

- Bump version in `cpu/__init__.py`
- Build source distribution: `python setup.py sdist`
- Check it looks OK, install it onto a virtualenv, run tests, run `cpu` as a tool
- Build release: `python setup.py sdist bdist_wheel`
- Push to PyPI: `twine upload dist/*`
- Test in a clean virtualenv that `pip install core-pytorch-utils` works with the new version
- Commit the version bump; add tag with `git tag v<VERSION_NUM>`; `git push --tags`

Happy hacking!