import tensorflow as tf
import numpy as np
from PIL import Image
import os
import xlwt
import model


def get_one_image(img_dir):
    image = Image.open(img_dir)
    image = image.resize([32, 32])
    image_arr = np.array(image)
    return image_arr


def test(test_file):
    log_dir = 'saves'
    image_arr = get_one_image(test_file)

    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 32, 32, 3])
        print(image.shape)
        p = model.mmodel(image, 1)
        logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32, shape=[32, 32, 3])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success')
                prediction = sess.run(logits, feed_dict={x: image_arr})
                max_index = np.argmax(prediction)
                print(max_index)
                return max_index
            else:
                print('No checkpoint')


DATA = []
i = 0
dir = 'D:/WCsPy/data/test/'
workbook = xlwt.Workbook(encoding='utf-8')
booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
for file_name in os.listdir(dir):
    i = i+1
    if i > 20:
        break
    print('Number:', i)
    label = str(test(dir+file_name))
    result = [file_name, label]
    DATA.append(result)
    print(DATA)

i = 0
for row in DATA:
    j = 0
    for col in row:
        booksheet.write(i, j, col)
        j = j + 1
    i = i + 1
workbook.save('result.xlsx')
