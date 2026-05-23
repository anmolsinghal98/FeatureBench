"""
Phase 2 scaffold: Claude Code agent driven by GitHub Spec Kit slash commands.

STATUS: NOT WIRED UP. This file is a placeholder so a future contributor can
implement the high-fidelity spec-driven variant without rediscovering the
shape of the integration.

For the lightweight A/B comparison today, prefer the prompt-wrapper approach
via `fb infer --agent claude_code --spec-mode ...` (see
`run_infer._inject_spec_kit_preamble`).

To activate this agent, after implementing the TODOs below, register it in
`featurebench/infer/agents/__init__.py` (add to imports, __all__, the `agents`
dict in `get_agent`) and in the `--agent` choices list in `run_infer.py`.

Design sketch
-------------
- `install_script`: reuse ClaudeCodeAgent.install_script, then additionally
  install Spec Kit:
      curl -fsSL https://astral.sh/uv/install.sh | sh
      uvx --from git+https://github.com/github/spec-kit specify-cli init \
          --here --ai claude --script sh --force
  (verify exact CLI flags against https://github.com/github/spec-kit)

- `get_run_command`: instead of one freeform `claude -p`, issue a *sequence*
  of non-interactive Claude Code calls so we can observe each Spec Kit phase
  in isolation. Sketch:

      claude -p '/specify <task>'   --output-format stream-json >> log
      claude -p '/plan'             --output-format stream-json >> log
      claude -p '/tasks'            --output-format stream-json >> log
      claude -p '/implement'        --output-format stream-json >> log

  Open question: whether `claude -p` preserves working-directory state across
  invocations well enough for slash commands to find their previous artifacts,
  or whether we need a single session driven via expect/pexpect. The wrapper
  approach (--spec-mode) sidesteps this entirely.

- `post_run_hook`: same success criteria as ClaudeCodeAgent, but read the
  *final* stream JSONL (the /implement run); also assert that
  `/testbed/.specify/{spec,plan,tasks}.md` exist in the container before
  declaring success.
"""

from featurebench.infer.agents.claude_code import ClaudeCodeAgent


class ClaudeCodeSpecKitAgent(ClaudeCodeAgent):
    """TODO: implement Phase 2 (real Spec Kit driver). See module docstring."""

    @property
    def name(self) -> str:
        return "claude_code_speckit"

    # TODO: override install_script to also install uv + spec-kit.
    # TODO: override get_run_command to drive /specify → /plan → /tasks → /implement.
    # TODO: extend post_run_hook to verify .specify/*.md artifacts exist.
