## Upgrading Procedure

New versions of kamodo's dependencies will require periodic updates to the
core package. The following steps should be followed to keep this package maintained.

1. Update github workflow
1. Create dockerfile
1. Update docker-compose.yml
1. update setup files
1. Update setup.cfg
1. Update pypi package

### Update github workflow

In the base of this repo `.github/workflows/kamodo-package.yml`, edit the jobs section to
reflect the inteded versions of python to support

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.7', '3.8', '3.9', '3.10', '3.11']
```

Whenever a commit is pushed to this repo, all of the listed python versions will be tested.
If any errors show up, this will reveal any changes we need so that all tests pass.

