# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2022-05-28 09:44:33
# @Last Modified by:   Your name
# @Last Modified time: 2022-05-28 15:57:40
#%%
# TensorFlow and tf.keras
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


#%%
# Read in data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Change color scale to black and white
train_images = train_images / 255.0
test_images = test_images / 255.0

# From finer to coarse layer
extra_layers = {
    'second_layer_label' : {
        'Tops' : ['T-shirt/top','Pullover','Shirt'],
        'Bottoms' : ['Trouser'],
        'Dresses' : ['Dress'],
        'Outers' : ['Coat'],
        'Shoes' : ['Sandal','Sneaker','Ankle boot'],
        'Accessories' : ['Bag']
        },
    'third_layer_label' : {
        'Clothes' : ['Tops','Bottoms','Dresses','Outers'], 
        'Goods' : ['Shoes','Accessories']
        }
}

#%%
# Add Hierarchical class labels to the training and test sets
def map_labels(data_labels):

    fine_dictionary = {
        0: 'T-shirt/top', 
        1: 'Trouser', 
        2: 'Pullover', 
        3: 'Dress', 
        4: 'Coat', 
        5: 'Sandal', 
        6: 'Shirt', 
        7: 'Sneaker', 
        8: 'Bag', 
        9: 'Ankle boot'
    }

    data_labels = pd.DataFrame(data_labels)
    
    data_labels = data_labels.rename(columns={data_labels.columns[0]: 'label'})
    data_labels['fine_label'] = data_labels['label'].map(fine_dictionary)
    
    # Mapped columns with extra layers added
    i = 'fine_label'
    for l in extra_layers:
        layer = {val:key for key, lst in extra_layers[l].items() for val in lst}
        data_labels[l] = data_labels[i].map(layer)
        i = l
    return data_labels


# Flattened data
def flattened_data(df, data_lables):
    x = df.shape[0]
    y1 = df.shape[1]
    y2 = df.shape[2]
    df_flattened = pd.DataFrame(df.flatten().reshape(x, y1*y2))
    df_full = pd.concat([map_labels(data_lables), df_flattened], axis=1)
    return df_flattened, df_full


#%% Add hierarchical 
train100_images_flattened, train100 = flattened_data(train_images, train_labels)
test_images_flattened, test = flattened_data(test_images, test_labels)




#%%
# Count number of extra layers added
print(extra_layers[len(extra_layers.keys())].items())


# %%
def model_coarse(df_withlabel, df_flattened):
    most_coarse_layer = list(extra_layers.keys())[-1]
    most_coarse_layer_cls = len(extra_layers[most_coarse_layer].items())

    model_coarse = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(most_coarse_layer_cls)
    ])

    model_coarse.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
                
    layer = {key:val for key, val in zip(extra_layers[most_coarse_layer].keys(), range(most_coarse_layer_cls))}
    model_course_labels = np.asarray(df_withlabel[most_coarse_layer].map(layer))

    model_coarse_start_time = datetime.now()
    model_coarse.fit(df_flattened, model_course_labels, epochs=10)
    model_coarse__end_time = datetime.now()
    model_coarse__train_time = model_coarse__end_time - model_coarse_start_time
    print("Model-Coarse Training Time = ",model_coarse__train_time)

# %%
model_coarse(train100, train100_images_flattened)

# %%
