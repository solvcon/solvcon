# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Check that agent skill directories stay in sync."""

import pathlib
import sys


AGENTS = {
    'codex': pathlib.Path('.agents/skills'),
    'claude': pathlib.Path('.claude/skills'),
    'cursor': pathlib.Path('.cursor/skills'),
}

# These skills replace functionality built into the other agents.
SKILL_AGENTS = {
    'code-review': {'codex'},
    'simplify': {'codex'},
    'verify': {'codex'},
}


def find_skills(skill_dir):
    """Return skill names and invalid entries in an agent skill directory."""
    skills = set()
    invalid = []
    for path in sorted(skill_dir.iterdir()):
        skills.add(path.name)
        if not path.is_dir():
            invalid.append(f'{path}: not a directory or broken symlink')
        elif not (path / 'SKILL.md').is_file():
            invalid.append(f'{path}: SKILL.md not found')
    return skills, invalid


def main():
    """Check skill presence, symlink targets, and configured exceptions."""
    found = {}
    failed = []

    for agent, skill_dir in AGENTS.items():
        if not skill_dir.is_dir():
            failed.append(f'{skill_dir}: skill directory not found')
            found[agent] = set()
            continue
        found[agent], invalid = find_skills(skill_dir)
        failed.extend(invalid)

    all_skills = set().union(*found.values())
    unknown = set(SKILL_AGENTS) - all_skills
    for skill in sorted(unknown):
        failed.append(f'{skill}: configured but not found for any agent')

    all_agents = set(AGENTS)
    for skill in sorted(all_skills):
        expected = SKILL_AGENTS.get(skill, all_agents)
        unexpected = {agent for agent in all_agents - expected
                      if skill in found[agent]}
        missing = {agent for agent in expected
                   if skill not in found[agent]}
        if missing:
            failed.append(f'{skill}: missing for {", ".join(sorted(missing))}')
        if unexpected:
            failed.append(
                f'{skill}: unexpected for {", ".join(sorted(unexpected))}'
            )
        if expected == all_agents and not missing:
            skill_files = [AGENTS[agent] / skill / 'SKILL.md'
                           for agent in sorted(expected)]
            if (
                all(path.is_file() for path in skill_files)
                and any(not skill_files[0].samefile(path)
                        for path in skill_files[1:])
            ):
                failed.append(f'{skill}: agent copies do not share a target')

    if failed:
        print(f'FAILED: {len(failed)} agent skill errors:')
        for error in failed:
            print(f'  - {error}')
        return 1

    print(f'SUCCESS: {len(all_skills)} skills match agent configuration.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
