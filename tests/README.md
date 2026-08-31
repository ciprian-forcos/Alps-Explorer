# Alps Explorer tests

The app is a single `index.html`. There is no bundler.

```
node tests/mobile-popup.spec.js
```

Serves this folder on an ephemeral `127.0.0.1` port (not github.io) and drives Chromium through `playwright-core`.

Optional env:

- `PLAYWRIGHT_CORE` — path to a `playwright-core` install
- `PLAYWRIGHT_CHROMIUM` — path to a Chrome/Chromium executable
- `ALPS_TEST_OUT` — JSON report path (default: temp dir)

Exit 0 only if every case passes.
