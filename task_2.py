import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.datasets import mnist
def load_data(normalize=False):
    (image_data,image_labels),(_,_) = mnist.load_data()
    X_train = image_data[:10000]
    y_train = image_labels[:10000]
    X_test = image_data[20000:21000]
    y_test = image_labels[20000:21000]
    
    if normalize:
        X_train = X_train/255.0
        X_test = X_test/255.0
    return X_train,y_train,X_test,y_test

def plot_training_error(training_loss,xlabel,ylabel,title):
    epochs = range(1,len(training_loss)+1)
    fig,ax = plt.subplots()
    ax.plot(epochs,training_loss)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.savefig('training_error_plot.png')
    plt.show()

def plot_test_accuracy(accuracy,xlabel,ylabel,title):
    epochs = range(1,len(accuracy)+1)
    fig,ax = plt.subplots()
    ax.plot(epochs,accuracy)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.savefig('test_accuracy_plot.png')
    plt.show()

def plot_accuracy_percentage(digits,dig_accuracy,xlabel,ylabel,title,name):
    fig,ax = plt.subplots()
    ax.bar(digits,dig_accuracy,width=0.95,align='center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for idx,val in enumerate(dig_accuracy):
        ax.text(idx-0.4,val-7, str(val),color = 'white',fontsize='10.4' ,fontweight='bold')  
     
    plt.savefig('digits_accuracy.png')
    plt.show()     

def create_model(hidden_nodes):
    model_X = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(hidden_nodes,activation='sigmoid'),
        tf.keras.layers.Dense(10,activation='softmax')
    ])

    return model_X


hidden_nodes = [10,35,100,300,500]

for nodes in hidden_nodes:
    learning_rate = 0.1
    X_train,y_train,X_test,y_test = load_data(normalize=True)

    model_0_1 = create_model(hidden_nodes=nodes)

    model_0_1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model_0_1.fit(X_train,y_train,epochs=200,verbose=3)

    training_loss_0_1 = model_0_1.history.history['loss']
    xlabel = 'Epochs'
    ylabel = 'Training Loss'
    title = 'Training Loss - Learning Rate 0.1 - Nodes - ' + str(nodes)
    # plot_training_error(training_loss_0_1,xlabel,ylabel,title)

    acc = model_0_1.history.history['accuracy']
    xlabel = 'Epochs'
    ylabel = 'Test Accuracy'
    title = 'Test Accuracy - Learning Rate 0.1'
    # plot_test_accuracy(acc,xlabel,ylabel,title)

    pred = np.argmax(model_0_1.predict(X_test),axis=1)
    test_accuracy =  np.sum(y_test == pred)/len(y_test)
    print("Test Accuracy With Learning Rate: {} is {}".format(learning_rate,test_accuracy))

    cm = confusion_matrix(pred,y_test)
    digits = list(range(10))
    dig_accuracy = []
    for num in digits:
        accuracy_per_digit = np.round(cm[num,num]/np.sum(cm[:,num])*100,1)
        # print(accuracy_per_digit)
        dig_accuracy.append(accuracy_per_digit)

    xlabel = 'Digits'
    ylabel = 'Accuracy(%)'
    title = 'Task1 - Accuracy Plot - learning rate - ' + str(learning_rate) + " Nodes " + str(nodes)
    name = 'task1_acc_plot_'+str(learning_rate) +" " + str(nodes) +'.jpg'
    plot_accuracy_percentage(digits,dig_accuracy,xlabel,ylabel,title,name) 
    
    
    
    learning_rate = 0.01
    X_train,y_train,X_test,y_test = load_data(normalize=True)


    model_0_01 = create_model(hidden_nodes=nodes)

    model_0_01.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model_0_01.fit(X_train,y_train,epochs=200,verbose=3)

    training_loss_0_01 = model_0_01.history.history['loss']
    xlabel = 'Epochs'
    ylabel = 'Training Loss'
    title = 'Training Loss - Learning Rate 0.01 - Nodes - ' + str(nodes)
    # plot_training_error(training_loss_0_01,xlabel,ylabel,title)

    acc = model_0_01.history.history['accuracy']
    xlabel = 'Epochs'
    ylabel = 'Test Accuracy'
    title = 'Test Accuracy - Learning Rate 0.01'
    # plot_test_accuracy(acc,xlabel,ylabel,title)

    pred = np.argmax(model_0_01.predict(X_test),axis=1)
    test_accuracy =  np.sum(y_test == pred)/len(y_test)
    print("Test Accuracy With Learning Rate: {} is {}".format(learning_rate,test_accuracy))

    cm = confusion_matrix(pred,y_test)
    digits = list(range(10))
    dig_accuracy = []
    for num in digits:
        accuracy_per_digit = np.round(cm[num,num]/np.sum(cm[:,num])*100,1)
        # print(accuracy_per_digit)
        dig_accuracy.append(accuracy_per_digit)

    xlabel = 'Digits'
    ylabel = 'Accuracy(%)'
    title = 'Task1 - Accuracy Plot - learning rate - ' + str(learning_rate) + " Nodes " + str(nodes)
    name = 'task1_acc_plot_'+str(learning_rate) +" " + str(nodes) +'.jpg'
    plot_accuracy_percentage(digits,dig_accuracy,xlabel,ylabel,title,name) 



    learning_rate = 0.001
    X_train,y_train,X_test,y_test = load_data(normalize=True)

    model_0_001 = create_model(hidden_nodes=nodes)

    model_0_001.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model_0_001.fit(X_train,y_train,epochs=200,verbose=3)

    training_loss_0_001 = model_0_001.history.history['loss']
    xlabel = 'Epochs'
    ylabel = 'Training Loss'
    title = 'Training Loss - Learning Rate 0.001 - Nodes - ' + str(nodes)
    # plot_training_error(training_loss_0_001,xlabel,ylabel,title)

    acc = model_0_001.history.history['accuracy']
    xlabel = 'Epochs'
    ylabel = 'Test Accuracy'
    title = 'Test Accuracy - Learning Rate 0.001'
    # plot_test_accuracy(acc,xlabel,ylabel,title)

    pred = np.argmax(model_0_001.predict(X_test),axis=1)
    test_accuracy =  np.sum(y_test == pred)/len(y_test)
    print("Test Accuracy With Learning Rate: {} is {}".format(learning_rate,test_accuracy))

    cm = confusion_matrix(pred,y_test)
    digits = list(range(10))
    dig_accuracy = []
    for num in digits:
        accuracy_per_digit = np.round(cm[num,num]/np.sum(cm[:,num])*100,1)
        # print(accuracy_per_digit)
        dig_accuracy.append(accuracy_per_digit)

    xlabel = 'Digits'
    ylabel = 'Accuracy(%)'
    title = 'Task1 - Accuracy Plot - learning rate - ' + str(learning_rate) + " Nodes " + str(nodes)
    name = 'task1_acc_plot_'+str(learning_rate) +" " + str(nodes) +'.jpg'
    plot_accuracy_percentage(digits,dig_accuracy,xlabel,ylabel,title,name) 