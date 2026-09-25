"""CLI unit tests: the job CLI and the umbrella CLI.

Thin, dqm-ml-shaped: no full job runs here (that is e2e's job). ``test_job_cli``
asserts argument handling, mocked-run ``execute``, the ``--save-config`` dump,
graceful YAML errors, and the kcai-registered ``run`` guards;
``test_umbrella_cli`` asserts the version/list/process dispatch and the
``optional_dependencies`` modes.
"""