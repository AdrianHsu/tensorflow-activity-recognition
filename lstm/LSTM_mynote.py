# All Includes

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics

import os

# Useful Constants
# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [ # 9個
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [ # 6個
    "WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING", 
    "STANDING", 
    "LAYING"
]
DATA_PATH = "data/"
DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"

# Preparing Dataset

TRAIN = "train/"
TEST = "test/"


# 先 Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []
    
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()
    
    return np.transpose(np.array(X_signals), (1, 2, 0))

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

# 使用 load_X
X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)


# 再來 Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]], 
        dtype=np.int32
    )
    file.close()
    
    # Substract 1 to each output class for friendly 0-based indexing 
    return y_ - 1

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

# 使用 load_y
y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

# 準備：Input Data 

training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each series)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series，一個series有 128 個 steps
n_input = len(X_train[0][0])  # 9 input parameters per timestep，一個step需要 9 個參數，就是INPUT_SIGNAL_TYPES的數量


# LSTM Neural Network's internal structure

n_hidden = 32 # Hidden layer num of features 自訂
n_classes = 6 # Total classes (should go up, or should go down)，就是 LABELS 個數

# Training 

learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset
batch_size = 1500
display_iter = 30000  # To show test set accuracy during training


# Some debugging info

print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
# (2947, 128, 9) (2947, 1) 0.0991399 0.395671
print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

# 一些輔助 functions, 用來 training
def LSTM_RNN(_X, _weights, _biases):
  

# height = 2
# 1 2 3 4
# 5 6 7 8
# 5 6 8 8 (3*4) 

# 9 12 13 14
# 15 16 17 18
# 5 6 7 8 (3*4)

# 這是個立體（3D)矩陣：(高h * 行row * 列col)
# x = [[[1,2,3,4],[5,6,7,8],[5,6,7,8]], [[9,12,13,14],[15,16,17,18],[5,6,7,8]]]   此3维数组为2x3x4，
# 可以看成是两个 3*4 的二维数组
# 对于二维数组，perm=[1,0,2], 1 代表三维数组的高（ 0->1 代表 height->row ）
# 0代表二维数组的行row（ 1->0 代表 row->height ），2代表二维数组的列col（ 2->2 代表 col 仍是 col ）
# y=tf.transpose(x, perm=[1,0,2])代表将 三位数组的高height和行row进行转置
# ie. (2,3,4) -> (3,2,4)
# 結果：y shape是 (3,2,4)
# [[[ 1  2  3  4], [ 9 12 13 14]], [[ 5  6  7  8], [15 16 17 18]], [[ 5  6  7  8], [ 5  6  7  8]]]

# height = 3
# 1 2 3 4
# 9 12 13 14 (2*4)

# 5 6 7 8 
# 15 16 17 18

# 5 6 7 8
# 5 6 7 8

# 範例 2
# y=tf.transpose(x, perm=[0, 2, 1])代表将 三位数组的row和col进行转置
# ie. (2,3,4) -> (2,4,3)
# 簡單！
# 1 5 5
# 2 6 6 ... (4*3)

    # input shape: (batch_size, n_steps, n_input)
    # 根據上面的話，是 (1500, 128, 9) -> (128, 1500, 9)
    # 所以也可用 tf.transpose(_X, [128, 1500, 9])  取代這行
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size

    # Reshape to prepare input to hidden activation
    # -1 就是懶得算的時候設定的ＸＤ
# tensor 't' is [[[1, 1, 1],
#                 [2, 2, 2]],
#                [[3, 3, 3],
#                 [4, 4, 4]],
#                [[5, 5, 5],
#                 [6, 6, 6]]] # shape (3*2*3)
# reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
#                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]  # shape(1*2*「9」), -1 的位置被帶入 9
# reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
#                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]  # shape(1*「2」*9), -1 的位置被帶入 2

	# 所以這就變成： (128, 1500, 9) -> (1, 128*1500 , 9)
    # new shape: (n_steps*batch_size, n_input)，三維變二維了
    _X = tf.reshape(_X, [-1, n_input])  # n_input = 9
    
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # 把資料切開：因為 rnn 都是吃小組一組input
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 
    # 所以這就變成： ( 128*1500 , 9 ) ->  128個 * ( 1500 , 9 )
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    # n_hidden 是 32，
    # 兩層用的參數一樣，weight也一樣
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # 把兩層basic包在一起，變成multiRNNCell
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    # 輸出成 static_rnn
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier, 
    # as in the image describing RNNs at the top of this page

    # 用 -1 就可以把它拉直成一維
    lstm_last_output = outputs[-1]
    
    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)
    # 把 1500 大小的 data 取出來用
    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s


def one_hot(y_):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


# 接著開始建立network

# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Graph weights
weights = {
	# 這就跟之前筆記的一樣
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# 參數設好後就丟進去function了
pred = LSTM_RNN(x, weights, biases)

# Loss, optimizer and evaluation
# 之前沒看過，是某種定義 loss 的方式
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the data

# 之前看過
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 開 train!

# 方便把每次的 loss 跟 accuracies記下來，待會做圖用
# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
# 有了 tf.InteractiveSession() 这样的交互式会话，它不需要用 sess.run(变量)”这种形式
# ，而是定义好会话对象后，每次执行tensor时，调用tensor.eval()即可。
# Config 先不管
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# 每次就跑一個 batch_size=1500 這麼多
# Perform Training steps with "batch_size" amount of example data at each loop
# training_iters他的值是多少？
# training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each series)
# training_iters = training_data_count * 300  # Loop 300 times on the dataset

step = 1
while step * batch_size <= training_iters:
    batch_xs = extract_batch_size(X_train, step, batch_size) # 一次 1500 筆
    batch_ys = one_hot(extract_batch_size(y_train, step, batch_size)) # 拉出 1500 筆的y之後，轉成1-hot

    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs, 
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        
        # To not spam console, show training accuracy/loss in this "if"
        print("Training iter #" + str(step*batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))
        
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy], 
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        # 加進去待會做圖用
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

# Training iter #2190000:   Batch Loss = 0.287106, Accuracy = 0.9700000286102295
# PERFORMANCE ON TEST SET: Batch Loss = 0.4670214056968689, Accuracy = 0.9216151237487793
# Optimization Finished!
print("Optimization Finished!")

# Accuracy for test data
# 開始測 test data 表現
one_hot_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

# 待會做圖用
test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))
# FINAL RESULT: Batch Loss = 0.45611169934272766, Accuracy = 0.9165252447128296

# 開始作圖！

# (Inline plots: )
# %matplotlib inline # inline 只能用在 jupyter

# 第一張圖：loss 下降圖
font = {
    'family' : 'Bitstream Vera Sans',
    'weight' : 'bold',
    'size'   : 18
}
matplotlib.rc('font', **font)


width = 12
height = 12
plt.figure(figsize=(width, height))

# train 的 axis 建起來， 大小照打就對了
indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
# 把 loss, accuracy 畫進去
plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

# 把 test建起來， 大小照打就對了
indep_test_axis = np.append(
    np.array( range(batch_size, len(test_losses)*display_iter, display_iter) [:-1]),
    [training_iters]
)
plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Test losses")
plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training iteration')

plt.show()


# 第二張圖： Confusion Matrix

predictions = one_hot_predictions.argmax(1)

# print("Testing Accuracy: {}%".format(100*accuracy))

# print("")
# print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
# print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
# print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

# print("")
# print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
# print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
# print("")
# print("Confusion matrix (normalised to % of total test data):")
# print(normalised_confusion_matrix)
# print("Note: training and testing data is not equally distributed amongst classes, ")
# print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

# Plot Results: 
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix, # 改這就好ＸＤ
    interpolation='nearest', 
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# end
sess.close()