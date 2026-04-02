# Tasks: Consolidate & Simplify Documentation (SSOT Architecture)

## Roadmap & Spec Phase

- [x] **Data Extraction**: Extract the complete 32-algorithm "Master Roadmap" from `CURRICULUM.md`.
- [x] **Graduate Roadmap to Spec**: Create a delta spec to merge the roadmap into `openspec/specs/algorithm-curriculum/spec.md`.
- [x] **Automate `algo-list.txt`**: Confirm if `algo-list.txt` is used by Any script. If not, delete it. If yes, replace it with a generation command.

## Rules & Standards Phase

- [x] **Simplify `.clinerules`**: Replace redundant detailed technical requirements with high-level pointers to OpenSpec files.
- [x] **Atomic Update Verification**: Confirm that the AI agent follows the pointers to the specifications.

## Root Document Phase

- [x] **Simplify `README.md`**: Strip out the redundant feature lists and point to specifications instead.
- [x] **Archive Overlaps**: Delete `CURRICULUM.md` after successful graduation to the spec.

## Sync & Archive Phase

- [x] **Sync Specs**: Execute `/opsx-sync` to finalize the main specification files.
- [x] **Verification**: Ensure all important information has been preserved in the specs.
- [x] **Archive**: Mark tasks as complete and run `/opsx-archive`.
