* clone ReFrame in your $HOME from https://github.com/eth-cscs/reframe.git

* Run with:

```
cd SPH-EXA_mini-app.git/scripts/reframe/

~/reframe.git/bin/reframe \
-C ~/reframe.git/config/cscs.py \
--skip-system-check --system daint:gpu \
--keep-stage-files \
--prefix=$SCRATCH \
--report-file $HOME/rpt.json \
--performance-report \
-v \
-r -c sqpatch.py
```

* Plot with:

```
cd SPH-EXA_mini-app.git/scripts/reframe/plot/
ln -fs $HOME/rpt.json .

python3 plot.py
```

png files will be written for xsmall, small, medium and large.

