# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 01:15:58 2020

@author: nianfd
"""

alpha = 0.999
print(alpha)
for global_step in range(0, 100, 0.5):
    #print(global_step)
    tempalpha = min(1 - 1 / (global_step + 1), alpha)
    print(tempalpha)