# Automated release system

This document explains how pyjanitor's automated release system works
and the design decisions behind it.
Understanding this system helps maintainers know
when releases happen and how to control versioning.

## Release philosophy

pyjanitor follows a continuous delivery approach to releases:

- **Patch releases** happen automatically on every merge to `dev`
- **Minor and major releases** are triggered manually by maintainers

The rationale is straightforward:
most merges are bug fixes, documentation improvements,
or small enhancements that don't require human decision-making
about version numbers.
By automating patch releases,
we reduce the friction of getting changes into users' hands
while preserving human control for significant version bumps.

## How automatic releases work

When code is merged to the `dev` branch, the following sequence occurs:

1. **Tests run first**: The `pyjanitor tests` workflow executes,
   running the full test suite across Python 3.11, 3.12, and 3.13.

2. **Release triggers on success**: If all tests pass,
   the `Auto-release` workflow automatically triggers
   via GitHub's `workflow_run` event.

3. **Duplicate check**: The workflow checks
   if the current commit already has a release tag.
   This prevents re-releasing
   when the workflow's own version bump commits trigger another test run.

4. **Version bump**: If no tag exists,
   `bump2version` increments the patch version
   (e.g., 0.32.3 becomes 0.32.4).

5. **Release notes generation**: The `llamabot` CLI generates release notes
   using an LLM, summarizing changes since the last release.

6. **Build and publish**: The package is built and published to PyPI
   using trusted publishing.

7. **GitHub release**: A GitHub release is created
   with the generated release notes.

The entire process is hands-off after merging.
Maintainers don't need to do anything for patch releases.

## Manual releases for minor and major versions

When a change warrants a minor or major version bump,
maintainers trigger the release manually:

**When to use minor version (e.g., 0.32.0 to 0.33.0):**

- New features that don't break existing functionality
- Significant enhancements to existing features
- New optional dependencies or modules

**When to use major version (e.g., 0.x.x to 1.0.0):**

- Breaking changes to the public API
- Removal of deprecated features
- Fundamental changes to how the library works

**How to trigger a manual release:**

1. Go to the [Actions tab][actions] in the repository
2. Select "Auto-release" from the workflows list
3. Click "Run workflow"
4. Choose `major`, `minor`, or `patch` from the dropdown
5. Click the green "Run workflow" button

[actions]: https://github.com/pyjanitor-devs/pyjanitor/actions

The workflow will run the same steps as an automatic release,
but with your chosen version bump type.

## Technical implementation

The release system uses GitHub Actions' `workflow_run` trigger,
which fires when another workflow completes.
Here's how the triggers work:

```yaml
on:
  # Automatic: triggers after tests complete on dev
  workflow_run:
    workflows: ["pyjanitor tests"]
    types:
      - completed
    branches:
      - dev

  # Manual: maintainer-triggered releases
  workflow_dispatch:
    inputs:
      version_name:
        type: choice
        options:
          - major
          - minor
          - patch
```

The job only runs when appropriate:

```yaml
if: |
  github.event_name == 'workflow_dispatch' ||
  (github.event.workflow_run.conclusion == 'success' &&
   github.event.workflow_run.event == 'push')
```

This condition ensures:

- Manual triggers always run
- Automatic triggers only run when tests *passed* (not failed)
- Automatic triggers only run for *push* events (not PR test runs)

## Preventing duplicate releases

When the auto-release workflow runs,
it creates a version bump commit and pushes it to `dev`.
This push triggers another test run,
which could trigger another release.
To prevent this infinite loop,
the workflow checks if the current commit already has a release tag:

```yaml
- name: Check if already released
  id: check_release
  run: |
    CURRENT_COMMIT=$(git rev-parse HEAD)
    if git tag --points-at $CURRENT_COMMIT | grep -q "^v"; then
      echo "already_released=true" >> $GITHUB_OUTPUT
    else
      echo "already_released=false" >> $GITHUB_OUTPUT
    fi
```

If a tag exists, all subsequent release steps are skipped.

## Troubleshooting

### Auto-release didn't trigger after merge

Check these common causes:

1. **Tests failed**: The release only triggers on successful test runs.
   Check the Actions tab for test failures.

2. **Not a push event**: Releases only trigger for pushes to `dev`,
   not for pull request test runs. This is intentional.

3. **Already released**: If the commit already has a tag,
   the release is skipped. Check `git tag --points-at HEAD`.

### Release failed mid-way

If a release fails after bumping the version but before publishing:

1. Check the Actions log to identify the failure point
2. If the tag was created but not pushed,
   you may need to manually push it or delete and recreate
3. If PyPI publish failed,
   you can re-run the failed job or trigger a manual release

### Wrong version was released

If you need to correct a version:

1. Do not try to overwrite an existing PyPI release
   (PyPI doesn't allow this)
2. Instead, create a new patch release with the fix
3. If the version number itself is wrong,
   update `.bumpversion.cfg` and `pyproject.toml` manually,
   commit, and trigger a new release

### How to skip a release for a specific merge

Currently, all merges to `dev` trigger a release.
If you need to merge something without releasing
(e.g., CI-only changes), you have two options:

1. Merge to a different branch first,
   then batch multiple changes into a single merge to `dev`
2. Cancel the auto-release workflow run manually
   in the Actions tab before it completes

**Note:** There's an open issue ([#1549][issue])
to consider adding filters for non-code changes.

[issue]: https://github.com/pyjanitor-devs/pyjanitor/issues/1549
