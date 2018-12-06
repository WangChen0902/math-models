import numpy as np
import tensorflow as tf
import os
import model
import inputData


def run_training():
    data_dir = 'D:/WCsPy/data/train/'
    log_dir = 'saves'
    image, label = inputData.get_files(data_dir)
    image_batches, label_batches = inputData.get_batches(image, label, 32, 32, 16, 20)
    print(image_batches.shape)
    p = model.mmodel(image_batches, 16)
    cost = model.loss(p, label_batches)
    train_op = model.training(cost, 0.001)
    acc = model.get_accuracy(p, label_batches)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(1000):
            print(step)
            if coord.should_stop():
                break
            _, train_acc, train_loss = sess.run([train_op, acc, cost])
            print("loss:{} accuracy:{}".format(train_loss, train_acc))
            if step % 100 == 0:
                check = os.path.join(log_dir, "model.ckpt")
                saver.save(sess, check, global_step=step)
    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


run_training()