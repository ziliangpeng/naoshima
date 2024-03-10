import sys
import lsm

source = sys.argv[1]
target = sys.argv[2]

with lsm.LSM(source) as source_lsm:
    with lsm.LSM(target) as target_lsm:
        for key, value in source_lsm.items():
            target_lsm[key] = value