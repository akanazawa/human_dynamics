#!/usr/bin/env python
import subprocess
import sys


def restart_until_success(cmd):
    ret_code = -1
    while ret_code != 0:
        ret_code = subprocess.call(cmd)


if __name__ == '__main__':
    cmd = sys.argv[1:]
    print ('Executing:', ' '.join(cmd))
    restart_until_success(cmd)
