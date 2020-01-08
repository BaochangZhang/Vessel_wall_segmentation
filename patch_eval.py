#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def evaluation_patch(start, end):
    for num in range(start, end):
        command_str = "python ./Eval.py "+str(num)
        os.system(command_str)


if __name__ == '__main__':
    evaluation_patch(0, 1063)