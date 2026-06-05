# Agent Skills

DeePMD-kit provides [Agent Skills](https://agentskills.io/what-are-skills) that
help AI agents run DeePMD-kit workflows in a reproducible way. These skills
capture project-specific operating knowledge—such as training inputs, model
deployment, LAMMPS integration, and Python inference patterns—so an agent can
turn a high-level request into concrete files, commands, and validation steps.

The DeePMD-kit skills are maintained in the [Computational Chemistry Agent
Skills](https://github.com/jinzhezenggroup/computational-chemistry-agent-skills)
repository, together with related skills for data preparation, molecular
dynamics, electronic-structure calculations, and workflow submission. Use the
current version from that repository when installing or citing the skills.

## DeePMD-kit skills

The repository currently includes the following DeePMD-kit-focused skills:

- `deepmd-train-dpa3`: Train DeePMD-kit models with the DPA3 descriptor and the
  PyTorch backend, including input generation, neighbor-selection choices,
  training, freezing, and testing.
- `deepmd-finetune-dpa3`: Fine-tune DPA3 models from self-trained checkpoints,
  multi-task pre-trained models, or built-in models downloaded by `dp
  pretrained download`.
- `deepmd-train-se-e2-a`: Train classical Deep Potential models with the
  `se_e2_a` descriptor, including preparation of training JSON files and
  post-training validation.
- `deepmd-python-inference`: Run Python and CLI inference with trained or
  frozen DeePMD-kit models, including energy, force, virial, descriptor, and
  model-deviation workflows.
- `lammps-deepmd`: Prepare, explain, and run LAMMPS simulations with DeePMD-kit
  potentials, including common NVE, NVT, and NPT setups.

Several companion skills are often useful in DeePMD-kit workflows:

- `dpdata-cli`: Convert structures, trajectories, and labeled datasets among
  common formats, including `deepmd/raw`, `deepmd/npy`, and `deepmd/hdf5`.
- `dpdata-driver`: Label `dpdata.System` objects through driver plugins such as
  DeePMD-kit, ASE, Gaussian, SQM, or hybrid drivers.
- `dpdisp-submit`: Submit shell commands and computational jobs to local
  machines or schedulers such as Slurm, PBS, LSF, and Bohrium.
- `packmol-generate-mixture`, `openbabel`, `rdkit-conf`, and related
  molecular-preparation skills: Build or convert starting structures before
  data generation or molecular dynamics.
- `lammps-reaxff`, `reacnetgenerator`, and electronic-structure skills: Prepare
  reference simulations, analyze reactive trajectories, or generate data for
  DeePMD-kit training.

See the [skills
summary](https://github.com/jinzhezenggroup/computational-chemistry-agent-skills#skills-summary)
for the full and most up-to-date list.

## Related reference

The computational-chemistry skills are described in the following paper:

- Mingwei Ding, Chen Huang, Yibo Hu, Yifan Li, Zitian Lu, Xingtai Yu, Duo
  Zhang, Wenxi Zhai, Tong Zhu, Qiangqiang Gu, and Jinzhe Zeng. [Automating
  Computational Chemistry Workflows via OpenClaw and Domain-Specific
  Skills](https://doi.org/10.48550/arXiv.2603.25522). arXiv:2603.25522, 2026.

## Install skills

### If you are a user

The easiest way is to send this page to your agent and ask it to install the
skills you need. In most cases, you do not need to copy skill files or run
installation commands yourself.

### If you are an agent

Install the skills from the latest version of
[jinzhezenggroup/computational-chemistry-agent-skills](https://github.com/jinzhezenggroup/computational-chemistry-agent-skills),
not from a stale local copy. One general installation route is:

1. Download the repository archive from
   <https://github.com/jinzhezenggroup/computational-chemistry-agent-skills/archive/refs/heads/master.zip>.
2. Unzip the archive.
3. From the OpenClaw workspace root, install each needed skill directory that
   contains a `SKILL.md` file, for example:

   ```bash
   npx -y skills add \
     $SKILLS_ROOT/machine-learning-potentials/deepmd-train-dpa3 \
     -a openclaw -y
   npx -y skills add \
     $SKILLS_ROOT/machine-learning-potentials/deepmd-finetune-dpa3 \
     -a openclaw -y
   npx -y skills add \
     $SKILLS_ROOT/machine-learning-potentials/deepmd-python-inference \
     -a openclaw -y
   npx -y skills add \
     $SKILLS_ROOT/molecular-dynamics/lammps-deepmd \
     -a openclaw -y
   ```

4. Start a new agent session so the installed skills are reloaded.

To install all visible skills from the repository, repeat the `skills add`
command for each top-level skill directory matching `*/*/SKILL.md`.

## Minimal verification

Ask the agent to perform a small task that exercises the installed skill
without launching an expensive calculation. For example:

- “Use the `deepmd-python-inference` skill to write a minimal Python snippet
  for loading a frozen DeePMD-kit model and evaluating one frame.”
- “Use the `deepmd-train-dpa3` skill to draft a small DPA3 training input for a
  water dataset, but do not start training.”
- “Use the `lammps-deepmd` skill to prepare an NVT LAMMPS input file for a
  DeePMD-kit model, and explain each command.”
