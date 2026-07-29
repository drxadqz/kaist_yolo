# Security policy

## Supported version

Security fixes are applied to the current default branch.

## Reporting

Please do not open a public issue for credentials, private dataset paths, or other
sensitive material. Contact the repository owner through the email address shown
on the owner's GitHub profile.

This repository never requires API keys for detector training or evaluation. The
optional experiment copilot reads `OPENAI_API_KEY` from the process environment;
it does not accept, persist, or print keys.

Before contributing, verify that the change contains no datasets, model weights,
database files, `.env` files, browser state, login artifacts, or machine-specific
absolute paths.
