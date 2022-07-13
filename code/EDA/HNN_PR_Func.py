# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2022-05-28 09:44:33
# @Last Modified by:   Your name
# @Last Modified time: 2022-05-30 12:35:47
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



# %%
# def model_coarse(train, train_flattened, test, test_flattened, layer_key):
#     most_coarse_layer = list(extra_layers.keys())[-1]
#     most_coarse_layer_cls = len(extra_layers[most_coarse_layer].items())

#     model_coarse = tf.keras.Sequential([
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(most_coarse_layer_cls)
#     ])

#     model_coarse.compile(optimizer='adam',
#                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy']
#     )
                
#     layer = {key:val for key, val in zip(extra_layers[most_coarse_layer].keys(), range(most_coarse_layer_cls))}
#     model_course_labels = np.asarray(train[most_coarse_layer].map(layer))

#     model_coarse_start_time = datetime.now()
#     model_coarse.fit(train_flattened, model_course_labels, epochs=10)
#     model_coarse__end_time = datetime.now()
#     model_coarse__train_time = model_coarse__end_time - model_coarse_start_time
#     print("Model-Coarse Training Time = ",model_coarse__train_time)

#     # Make prediction on test set
#     model_coarse_probbability_model = tf.keras.Sequential([model_coarse, tf.keras.layers.Softmax()])
#     model_coarse_predictions = model_coarse_probbability_model.predict(test_flattened)
#     model_coarse_prediction_label = np.argmax(model_coarse_predictions, axis=1)
    
#     # Add labels mapped to predictions
#     most_coarse_layer_pred = most_coarse_layer + '_prediction'
#     model_coarse_prediction_labels = pd.DataFrame(model_coarse_prediction_label)
#     model_coarse_prediction_labels = model_coarse_prediction_labels.rename(columns={model_coarse_prediction_labels.columns[0]: most_coarse_layer_pred})

#     layer_mapped = {val:key for key, val in zip(extra_layers[most_coarse_layer].keys(), range(most_coarse_layer_cls))}
#     model_coarse_prediction_labels[most_coarse_layer_pred]= model_coarse_prediction_labels[most_coarse_layer_pred].map(layer_mapped)

#     # print(model_coarse_prediction_labels.head(10))

#     # Merge Coarse predictions back to test dataframe
#     test = pd.concat([test, model_coarse_prediction_labels], axis=1)

#     # Split dataframe into sub-category (clothes and goods, in this case)
#     d = {name: pd.DataFrame() for name in extra_layers[most_coarse_layer].keys()}

#     for name in test[most_coarse_layer_pred].unique():
#         d[name] = test[(test[most_coarse_layer_pred] == name)]
    
#     return test, d
        
# #%%
# test, coarse = model_coarse(train100, train100_images_flattened, test, test_images_flattened)        


#%%
def model_for_each_layer(train, train_flattened, test, test_flattened, layer_key):
    most_coarse_layer = layer_key
    most_coarse_layer_cls = len(extra_layers[most_coarse_layer].items())

    model_coarse = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(most_coarse_layer_cls)
    ])

    model_coarse.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
    )
                
    layer = {key:val for key, val in zip(extra_layers[most_coarse_layer].keys(), range(most_coarse_layer_cls))}
    model_course_labels = np.asarray(train[most_coarse_layer].map(layer))

    model_coarse_start_time = datetime.now()
    model_coarse.fit(train_flattened, model_course_labels, epochs=10)
    model_coarse__end_time = datetime.now()
    model_coarse__train_time = model_coarse__end_time - model_coarse_start_time
    print("Model-Coarse Training Time = ",model_coarse__train_time)

    # Make prediction on test set
    model_coarse_probbability_model = tf.keras.Sequential([model_coarse, tf.keras.layers.Softmax()])
    model_coarse_predictions = model_coarse_probbability_model.predict(test_flattened)
    model_coarse_prediction_label = np.argmax(model_coarse_predictions, axis=1)
    
    # Add labels mapped to predictions
    most_coarse_layer_pred = most_coarse_layer + '_prediction'
    model_coarse_prediction_labels = pd.DataFrame(model_coarse_prediction_label)
    model_coarse_prediction_labels = model_coarse_prediction_labels.rename(columns={model_coarse_prediction_labels.columns[0]: most_coarse_layer_pred})

    layer_mapped = {val:key for key, val in zip(extra_layers[most_coarse_layer].keys(), range(most_coarse_layer_cls))}
    model_coarse_prediction_labels[most_coarse_layer_pred]= model_coarse_prediction_labels[most_coarse_layer_pred].map(layer_mapped)

    # print(model_coarse_prediction_labels.head(10))

    # Merge Coarse predictions back to test dataframe
    test = pd.concat([test, model_coarse_prediction_labels], axis=1)


    # Split dataframe into sub-category (clothes and goods, in this case)
    train_split = {name: pd.DataFrame() for name in extra_layers[most_coarse_layer].keys()}
    test_split = {name: pd.DataFrame() for name in extra_layers[most_coarse_layer].keys()}

    for name in test[most_coarse_layer_pred].unique():
        train_split[name] = test[(test[most_coarse_layer_pred] == name)]
        test_split[name] = test[(test[most_coarse_layer_pred] == name)]

    
    return train_split, test_split


#%%
def flattened_data(df, data_lables):
    x = df.shape[0]
    y1 = df.shape[1]
    y2 = df.shape[2]
    df_flattened = pd.DataFrame(df.flatten().reshape(x, y1*y2))
    df_full = pd.concat([map_labels(data_lables), df_flattened], axis=1)
    return df_flattened, df_full

#%%
# fl_train = train100
# fl_train_flattened = train100_images_flattened
# fl_test = test
# fl_test_flattened = test_images_flattened
# next_model_df = {}

for j in range(len(extra_layers), 1, -1):
    layer_key = list(extra_layers.keys())[j-1]
    # train_split, test_split = model_for_each_layer(fl_train, fl_train_flattened, fl_test, fl_test_flattened,layer_key)
    for k, train, test in zip(extra_layers[layer_key], train_split, test_split):
        print(extra_layers[layer_key][k])
        # print(train_split[train].head())
        # print(test_split[test].head())
    
    # fl_train = {name: pd.DataFrame() for name in extra_layers[most_coarse_layer].keys()}

    # for k, name in zip(extra_layers[layer_key],d):
    #     print(extra_layers[layer_key][k])
    #     print(d[name].head())
        

    # k = list(extra_layers.keys())[j-1]
    # for i in extra_layers[k].keys():
    #     print(i)
#%%
for train, test in zip(train_split, test_split):
    print(train_split[train].head())
    print(test_split[test].head())

#%%
for j in range(len(extra_layers), 1, -1):
    layer_key = list(extra_layers.keys())[j-1]
    # coarse = model_for_each_layer(train100, train100_images_flattened, test, test_images_flattened,layer_key)
    for k, name in zip(extra_layers[layer_key],coarse):
        print(k)
        print(extra_layers[layer_key][k])
        print(coarse[name].head())

# %%
d = model_coarse(train100, train100_images_flattened, test, test_images_flattened)

# %%
coarse_group = model1.groupby(model1.third_layer_label_prediction)
#%%
for j in extra_layers[2].items():
    print(j)
# %%
