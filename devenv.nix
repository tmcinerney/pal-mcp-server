{ pkgs, lib, ... }:
# PAL MCP Server development environment
#
# Usage:
#   1. Install devenv: https://devenv.sh/getting-started/
#   2. Run `devenv shell` (or use direnv with the included .envrc)
#   3. Set API keys via environment, .env file, or SecretSpec provider
#   4. Run `./code_quality_checks.sh` to verify setup
#
# SecretSpec (optional):
#   Create devenv.local.yaml to configure your secret provider:
#     secretspec:
#       enable: true
#       provider: "onepassword://YourVault"
{
  languages.python = {
    enable = true;
    version = lib.strings.trim (builtins.readFile ./.python-version);
    uv = {
      enable = true;
      sync = {
        enable = true;
        allGroups = false;
        groups = [ "dev" ];
      };
    };
    venv.enable = true;
  };

  # Lint tools provided via Nix to avoid dynamically linked binary issues
  # on NixOS. Split into a separate dependency-group in pyproject.toml -
  # on non-NixOS platforms, `uv sync --group lint` works too.
  packages = with pkgs; [
    ruff
    black
    isort
    pre-commit
  ];

  env = {
    DEFAULT_MODEL = "auto";
    LOG_LEVEL = "DEBUG";
  };
}
