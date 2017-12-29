#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:22:45 2017

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 7
AB_Improved = (10, 7, 8, 7, 7, 6, 5)
Custom_Agent_1 = (9, 7, 8, 4, 5, 5, 5)
Custom_Agent_2 = (10, 6, 9, 7, 5, 7, 2)
Custom_Agent_3 = (10, 7, 9, 8, 6, 7, 5)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 1
opacity = 0.8
 
rects1 = plt.bar(1.5*index, AB_Improved, 0.25*bar_width,
                 alpha=opacity,
                 color='b',
                 label='AB_Improved')
 
rects2 = plt.bar(1.5*index + 0.25*bar_width, Custom_Agent_1, 0.25*bar_width,
                 alpha=opacity,
                 color='g',
                 label='Custom_Agent_1')

rects3 = plt.bar(1.5*index + 0.5*bar_width, Custom_Agent_2, 0.25*bar_width,
                 alpha=opacity,
                 color='r',
                 label='Custom_Agent_2')

rects4 = plt.bar(1.5*index + 0.75*bar_width, Custom_Agent_3, 0.25*bar_width,
                 alpha=opacity,
                 color='g',
                 label='Custom_Agent_3')
 
plt.xlabel('Opponent Agent')
plt.ylabel('Wins (out of 10)')
plt.title('Comparison of Different Agent Performances in Isolation Tournament')
plt.xticks(1.5*index, ('Rand', 'MMO', 'MMC', 'MMI', 'ABO', 'ABC', 'ABI'))
plt.legend(fontsize='small')
 
plt.tight_layout()
plt.show()
fig.savefig('agents.png')