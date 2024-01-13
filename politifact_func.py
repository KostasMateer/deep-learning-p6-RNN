from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import json

def json_loader(file):
    statements = []
    values = []
    
    f = open(file, 'r')
    lines = f.readlines()
    for line in lines:
        statements.append(json.loads(line)['statement'])
        verdict = json.loads(line)['verdict']
        if (verdict == 'pants-fire'):
            values.append(0)
        elif (verdict == 'false'):
            values.append(1)
        elif (verdict == 'mostly-false'):
            values.append(2)
        elif (verdict == 'half-true'):
            values.append(3)
        elif (verdict == 'mostly-true'):
            values.append(4)
        elif (verdict == 'true'):
            values.append(5)
    
    f.close()
    return statements, values

def process_data(statements, values, classes):
    X_train, X_test, y_train, y_test = train_test_split(statements, values, test_size=0.2)
    
    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)
    
    X_train = tf.convert_to_tensor(X_train, dtype=tf.string)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.string)
    
    return X_train, X_test, y_train,y_test