# A(square) Rock Gene Essentiality Hackathon

Gene essentiality hackathon for A(square) Rock team.


## Installation

### Clone repository

```shell
git clone --recurse-submodules https://github.com/otchita-owkin/asquarerock-hackathon
cd asquarerock-hackathon/
```

### Create environement

Create new conda environement using:

```shell
conda create -q --prefix /workspace/envs/hackathon_env python=3.8 -y
conda activate /workspace/envs/hackathon_env
```
Then add it to notebook kernel using:

```shell
python -m ipykernel install --user --name=hackathon_env
```

### Install all dependencies

```shell
make install
```
