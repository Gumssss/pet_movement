# API configuration

The local `openai.json` contains the API key, base URL, and model. It is Git-ignored. Use `openai.example.json` as the template for local setup.

The Sol result set is available in [outputs/vlm/api/openai](../outputs/vlm/api/openai/gpt6_1_sol_sixplot_batch/full_test/README.md). The archived run's cancellation marker remains in the batch directory and prevents submission of that batch. Its original request bundles are listed in the [cleanup manifest](../outputs/records/cleanup_manifest.json).

Install `scripts/requirements_openai_batch.txt` in the `collar` environment before making API calls.
