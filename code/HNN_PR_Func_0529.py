# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2022-05-28 09:44:33
# @Last Modified by:   Your name
# @Last Modified time: 2022-05-30 21:51:07
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


#%%
# From finer to coarse layer
# Only allows 2 extra layers
extra_layers = {
        'Clothes' : {'Tops' : ['T-shirt/top','Pullover','Shirt'],
                     'Bottoms' : ['Trouser'],
                     'Dresses' : ['Dress'],
                    'Outers' : ['Coat']
        }, 
        'Goods' : {'Shoes' : ['Sandal','Sneaker','Ankle boot'],
                   'Accessories' : ['Bag']
        }
}


#%%
# Make each layer a dict
def layers_dict(dict, key, val):
    if key in dict.keys():
        dict[key].append(val) # append the new number to the existing array at this slot
    else:
        dict[key] = [val] # create a new array in this slot
    return dict


# Map to data label
def layers_mapped(mapped_dict, mapped_to_df, col_name):
    layer = {val:key for key, lst in mapped_dict.items() for val in lst}
    last_col = list(mapped_to_df.columns)[-1]
    mapped_to_df[col_name] = mapped_to_df[last_col].map(layer)
    return mapped_to_df


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

    coarse_dict = {}
    medium_dict = {}
    # fine_dict = {}
    for k in extra_layers:
        for l in extra_layers[k]:
            coarse_dict = layers_dict(coarse_dict, k, l)
            try:
                for m in extra_layers[k][l]:
                    medium_dict = layers_dict(medium_dict, l, m)
                    # for n in extra_layers[k][l][m]:
                    #     fine_dict = layers_dict(fine_dict, m, n)
            except: pass
            
    medium_mapped = layers_mapped(medium_dict, data_labels,'medium_label')
    coarse_mapped = layers_mapped(coarse_dict, medium_mapped,'coarse_label')
    return coarse_mapped


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
#        
def model_for_each_layer(train, test, dict, layer_name):
    most_coarse_layer = dict
    most_coarse_layer_cls = len(dict)
    print(most_coarse_layer)
    print(most_coarse_layer_cls)
    model_coarse = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(most_coarse_layer_cls)
    ])

    model_coarse.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
    )
                
    layer = {key:val for key, val in zip(dict, range(most_coarse_layer_cls))}
    print(layer)
    model_course_labels = np.asarray(train[layer_name].map(layer))
    print(model_course_labels)

    model_coarse_start_time = datetime.now()
    train_flattened = train.iloc[:,4:788]
    model_coarse.fit(train_flattened, model_course_labels, epochs=10)
    model_coarse__end_time = datetime.now()
    model_coarse__train_time = model_coarse__end_time - model_coarse_start_time
    print("Model-Coarse Training Time = ",model_coarse__train_time)

    # Make prediction on test set
    test_flattened = test.iloc[:,4:788]
    model_coarse_probbability_model = tf.keras.Sequential([model_coarse, tf.keras.layers.Softmax()])
    model_coarse_predictions = model_coarse_probbability_model.predict(test_flattened)
    model_coarse_prediction_label = np.argmax(model_coarse_predictions, axis=1)
    print(model_coarse_prediction_label)
    
    # Add labels mapped to predictions
    most_coarse_layer_pred = layer_name + '_prediction'
    model_coarse_prediction_labels = pd.DataFrame(model_coarse_prediction_label)
    model_coarse_prediction_labels = model_coarse_prediction_labels.rename(columns={model_coarse_prediction_labels.columns[0]: most_coarse_layer_pred})

    layer_mapped = {val:key for key, val in zip(dict, range(most_coarse_layer_cls))}
    model_coarse_prediction_labels[most_coarse_layer_pred] = model_coarse_prediction_labels[most_coarse_layer_pred].map(layer_mapped)

    # Merge Coarse predictions back to test dataframe
    test = test.join(model_coarse_prediction_labels)
    
    # Split dataframe into sub-category (clothes and goods, in this case)
    try:
        train_split = {name: pd.DataFrame() for name in dict.keys()}
        test_split = {name: pd.DataFrame() for name in dict.keys()}


        for name in test[most_coarse_layer_pred].unique():
            train_split[name] = train[(train[layer_name] == name)]
            test_split[name] = test[(test[most_coarse_layer_pred] == name)]

    except AttributeError:
        train_split = train
        test_split = test
    
    return train_split, test_split



#%%
# Coarse label
coarse_train, coarse_test = model_for_each_layer(train100, test, extra_layers, 'coarse_label')


#%%
# Medium label
medium_train = {name: pd.DataFrame() for name in extra_layers.keys()}
medium_test = {name: pd.DataFrame() for name in extra_layers.keys()}

for k in extra_layers:
    medium_train[k], medium_test[k] = model_for_each_layer(
        coarse_train[k], coarse_test[k], extra_layers[k], 'medium_label')


#%%
# Fine label
fine_train = {name: pd.DataFrame() for name in extra_layers.keys()}
fine_test = {name: pd.DataFrame() for name in extra_layers.keys()}

for k in extra_layers:
    fine_train[k] = {name: pd.DataFrame() for name in extra_layers[k].keys()}
    fine_test[k] = {name: pd.DataFrame() for name in extra_layers[k].keys()}

for k in extra_layers:
    for l in extra_layers[k]:
        fine_train[k][l], fine_test[k][l] = model_for_each_layer(
            medium_train[k][l], medium_test[k][l], extra_layers[k][l], 'fine_label')


#%%
# Append fine_test dataframe back together
final_test = pd.DataFrame(columns=list(medium_test['Goods']['Shoes'].columns))
for k in extra_layers:
    for l in extra_layers[k]:
        final_test = pd.concat([final_test, fine_test[k][l]])
        print(final_test.shape)

final_test = final_test[final_test['fine_label'].notnull()]
# final_test = pd.concat([final_test, medium_test['Goods']['Shoes'], medium_test['Goods']['Accessories']])

# final_test = pd.DataFrame(columns=list(coarse_test['Clothes'].columns))
# final_test = pd.concat([final_test, coarse_test['Clothes'], coarse_test['Goods']])
# print(final_test)
# for k in extra_layers:
#     for l in extra_layers[k]:
#         final_test.append(fine_test[k][l])
        # print(fine_test[k][l])
        # print('******')

# %%