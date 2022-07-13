# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2022-05-28 09:44:33
# @Last Modified by:   Your name
# @Last Modified time: 2022-05-30 21:51:07
#%%
# TensorFlow and tf.keras
from numpy.core.fromnumeric import shape
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
        'Clothes' : {'Shirts' : ['T-shirt/top','Shirt'],
                    'Other Tops' : ['Pullover', 'Dress', 'Coat'],
                     'Bottoms' : ['Trouser']
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

    earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=10,
    verbose=3,
    mode='auto',
    baseline=None,
    restore_best_weights=True
    )

    from tensorflow.keras.callbacks import TensorBoard
    from time import time
    tb = TensorBoard(log_dir=f"logs\\{time()}")
                
    layer = {key:val for key, val in zip(dict, range(most_coarse_layer_cls))}
    print(layer)
    model_course_labels = np.asarray(train[layer_name].map(layer))
    print(model_course_labels)

    model_coarse_start_time = datetime.now()
    train_flattened = train.iloc[:,4:788]
    model_coarse.fit(train_flattened, model_course_labels, validation_split=0.1, epochs=200, callbacks=[earlystop, tb])
    model_coarse__end_time = datetime.now()
    model_coarse__train_time = model_coarse__end_time - model_coarse_start_time
    print("Model-Coarse Training Time = ", model_coarse__train_time)

    # Make prediction on test set
    test_flattened = test.iloc[:,4:788]
    model_coarse_probbability_model = tf.keras.Sequential([model_coarse, tf.keras.layers.Softmax()])
    model_coarse_predictions = model_coarse_probbability_model.predict(test_flattened)
    model_coarse_prediction_label = np.argmax(model_coarse_predictions, axis=1)
    print(model_coarse_prediction_label)
    
    # Add labels mapped to predictions
    most_coarse_layer_pred = layer_name + '_prediction'

    test[most_coarse_layer_pred] = model_coarse_prediction_label

    layer_mapped = {val:key for key, val in zip(dict, range(most_coarse_layer_cls))}
    test[most_coarse_layer_pred] = test[most_coarse_layer_pred].map(layer_mapped)

    # Split dataframe into sub-category (clothes and goods, in this case)
    try:
        train_split = {name: pd.DataFrame() for name in dict.keys()}
        test_split = {name: pd.DataFrame() for name in dict.keys()}

    except AttributeError:
        train_split = {name: pd.DataFrame() for name in dict}
        test_split = {name: pd.DataFrame() for name in dict}

    for name in test[most_coarse_layer_pred].unique():
        train_split[name] = train[(train[layer_name] == name)]
        test_split[name] = test[(test[most_coarse_layer_pred] == name)]
    
    return train_split, test_split


#%%
# Coarse label
coarse_train, coarse_test = model_for_each_layer(train100, test, extra_layers, 'coarse_label')

#%%
# Medium label
medium_train = {name: pd.DataFrame() for name in extra_layers.keys()}
medium_test = {name: pd.DataFrame() for name in extra_layers.keys()}
t1 = {name: pd.DataFrame() for name in extra_layers.keys()}
t2 = {name: pd.DataFrame() for name in extra_layers.keys()}

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
fine_final_test = pd.DataFrame(columns=list(fine_test['Goods']['Shoes']['Sandal'].columns))
for k in extra_layers:
    for l in extra_layers[k]:
        for m in extra_layers[k][l]:
            fine_final_test = pd.concat([fine_final_test, fine_test[k][l][m]])
            print(k,l,m,fine_final_test.shape, fine_test[k][l][m].shape)

#%%
medium_final_test = pd.DataFrame(columns=list(medium_test['Goods']['Shoes'].columns))
for k in extra_layers:
    for l in extra_layers[k]:
        medium_final_test = pd.concat([medium_final_test, medium_test[k][l]])
        print(k,l,medium_final_test.shape, medium_test[k][l].shape)

#%%
coarse_final_test = pd.DataFrame(columns=list(coarse_test['Goods'].columns))
for k in extra_layers:
    coarse_final_test = pd.concat([coarse_final_test, coarse_test[k]])
    print(k,coarse_final_test.shape, coarse_test[k].shape)

#%%
fine_final_test.head()
# %%
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print("Fine Level Accuracy = ", accuracy_score(fine_final_test.fine_label, fine_final_test.fine_label_prediction))
print(classification_report(fine_final_test.fine_label, fine_final_test.fine_label_prediction))
# %%
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print("Medium Level Accuracy = ", accuracy_score(fine_final_test.medium_label, fine_final_test.medium_label_prediction))
print(classification_report(fine_final_test.medium_label, fine_final_test.medium_label_prediction))
# %%
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print("Coarse Level Accuracy = ", accuracy_score(fine_final_test.coarse_label, fine_final_test.coarse_label_prediction))
print(classification_report(fine_final_test.coarse_label, fine_final_test.coarse_label_prediction))

# %%
