# Contributing

## Creating Issues

For bugs, open an issue following [this guide](https://github.com/EnsembleGovServices/kamodo-core/blob/joss/.github/ISSUE_TEMPLATE/bug_report.md).

For features, provide enough context to illustrate the intended use case. If possible, include a test function that should pass once the feature is considered complete.


## Developer setup

Clone the git repo and install kamodo in developer mode: from the base of the repo:

```sh
pip install -e .
```

!!! note
    If you add any files while in editor mode, make sure they will be picked up by the MANIFEST.in file.

Kamodo is cross-platform, so you should be able work on whatever developing environment is convenient for you.
However, we find it helpful to mount the code into a docker container and we provide a docker-compose.yaml. This causes the least disruption with your machine and makes it easier to deploy Kamodo containers to cloud services.
If you have installed docker with docker `compose`, you can spin up a developer environment with one line from the base of the repo:

```sh
docker compose up
```


## Creating a Pull Request

Want to contribute to Kamodo-core? Open a pull request!

### Branching

First create a branch off of `master`, named after the feature or issue you are targeting.

```sh
git checkout -b issue-999
```

You may start your PR when you prefer, depending on how much support you'll need to complete it, but remember it should be up-to-date with `master` before it can be merged.


### Testing

Run tests locally prior to pushing.

```sh
python -m pip install flake8 pytest
pip install pytest-cov
```

Then, from the base of the git repo

```sh
pytest --cov kamodo.kamodo --cov kamodo.util --cov plotting kamodo/test_plotting.py kamodo/test_kamodo.py kamodo/test_utils.py
```

Try to at least maintain the current code coverage with your PR.


### Hourly (optional)

Consider using [hourly](https://github.com/asherp/hourly) for time tracking your branch. This will allow future developers to see which parts of the code receive the most attention. 

If you're using hourly, you'll want to configure `hourly.yaml` so that your worklog points to a file named after you:

```yaml
work_log:
  header_depth: 3
  filename: worklogs/my_github_username.md
```




