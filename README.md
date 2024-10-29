# EBM package 

This repository contains contents for the python package of doing Event-Based modelling. 

## CHTC

Create a virtual environment

```sh
python3 -m venv ebm
source ebm/bin/activate
pip install -r requirements.txt
```

zip folders:

```sh
tar -czf ebm.tar.gz ebm/
tar -czf data.tar.gz data/
```

Check disk space:

```sh
du -h --max-depth=1 ~/ebm_package
```

These two files are too large to be uploaded to github. So you need to generate them on your local branch before running codes below. 

```sh
bash condor.sh
```

