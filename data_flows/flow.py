#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib.image as mpimg
import random
import math
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "causal_data" / "flow_noise"
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

count=0
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

for r in range(5, 35):
    for h_raw in range(10, 40):
        for hole in range(6, 15):
            # r = abs(np.random.randn() * 10)
            r += np.random.randn() * 0.1
            # hole += np.random.uniform()
            hole += r*0.5 + 6
            ball_r = r/30.0
            # h = pow(ball_r,3)+np.random.randn() + 4 #h_raw/10.0 #+ np.random.uniform()
            h = pow(ball_r,3)+h_raw/10.0 #+ np.random.uniform()
            deep = hole/3.0
            plt.rcParams['figure.figsize'] = (1.0, 1.0)
            ax = plt.gca()
            # water in cup 
            rect = plt.Rectangle(([3, 0]),5,5+h,color='lightskyblue')
            ax.add_artist(rect)
            ball = plt.Circle((5.5,+ball_r+0.5), ball_r, color = 'firebrick')
            ## cup
            left = plt.Polygon(([3, 0],[3, 19]), color = 'black', linewidth = 2)
            right_1 = plt.Polygon(([8, 0],[8, deep]), color = 'black', linewidth = 2)
            right_2 = plt.Polygon(([8, deep+0.4],[8, 19]), color = 'black', linewidth = 2)
            ax.add_artist(left)
            ax.add_artist(right_1)
            ax.add_artist(right_2)
            ax.add_artist(ball)
    
            #water line
            y = np.linspace(deep,0.5)
            epsilon = 0.01 * np.max([np.abs(np.random.randn()),1])
            tmp = h*deep
            x = np.sqrt(2*(0.98+epsilon)*h*(deep-y))+8
            x_max = x[-1]-8
            x_true = np.sqrt(2*(0.98 )*h*(deep-0.5))
            plt.plot(x,y,color='lightskyblue',linewidth = 2)
    
            ##ground
            x = np.linspace(0,20,num=50)
            y = np.zeros(50)+0.2
            plt.plot(x,y,color='black',linewidth = 2)
            
            ax.set_xlim((0, 20))
            ax.set_ylim((0, 20))
    
            plt.axis('off')
            # Temporary save the colored image
            if count == 4:
                filename = TEST_DIR / f"a_{r}_{h}_{tmp}_{hole}.png"
                count = 0
            else:
                filename = TRAIN_DIR / f"a_{r}_{h}_{tmp}_{hole}.png"
                count += 1

            plt.savefig(filename, dpi=96)
            plt.clf()

            # # Convert the image to grayscale
            # with Image.open(filename) as img:
            #     gray_img = img.convert("L")
            #     gray_img.save(filename)
