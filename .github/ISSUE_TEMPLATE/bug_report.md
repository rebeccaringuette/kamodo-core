---
name: Bug report
about: Create a report to help us improve Kamodo!
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
If the problem arises while registering/plotting a function using a kamodo object, set the kamodo object's verbose flag and copy and paste the output you get when you register:

```python
k = Kamodo(verbose=True)
# or, if you have already initialized an object, set the verbose flag like this:
# k.verbose = True
k['f'] = ... # expression or function implementation here
... # verbose output followed by error
```

```console
< copy and paste verbose output and error logs here >
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem. If there's something wrong with a plot, use the `export png` button in the plotly-generated figure.


**Additional context**
Add any other context about the problem here.
