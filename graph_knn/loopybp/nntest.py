import tensorflow as tf
from utils.nn import linearND, linear
from mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list
import math, sys, random
from optparse import OptionParser
from functools import partial
        
parser = OptionParser()
parser.add_option("-e", "--test", dest="test_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-b", "--batch", dest="batch_size", default=20)
parser.add_option("-w", "--hidden", dest="hidden_size", default=265)
parser.add_option("-d", "--depth", dest="depth", default=4)
opts,args = parser.parse_args()

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
depth = int(opts.depth) - 1

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

input_atom = tf.placeholder(tf.float32, [batch_size, None, adim])
input_bond = tf.placeholder(tf.float32, [batch_size, None, adim+bdim])
atom_graph = tf.placeholder(tf.float32, [batch_size, None, None])
bond_graph = tf.placeholder(tf.float32, [batch_size, None, None])
node_mask = tf.placeholder(tf.float32, [batch_size, None])
src_holder = [input_atom, input_bond, atom_graph, bond_graph, node_mask]
label = tf.placeholder(tf.float32, [batch_size])
node_mask = tf.expand_dims(node_mask, -1)

with tf.variable_scope("encoder"):
    binput = linearND(input_bond, hidden_size, "bond_embedding", init_bias=None)
    message = tf.nn.relu(binput)
    with tf.variable_scope("loopybp") as scope:
        for i in xrange(depth):
            nei_message = linearND(tf.batch_matmul(bond_graph, message), hidden_size, "bp", init_bias=None)
            message = tf.nn.relu(binput + nei_message)
            scope.reuse_variables()

    ainput = linearND(input_atom, hidden_size, "atom_embedding", init_bias=None)
    nei_message = linearND(tf.batch_matmul(atom_graph, message), hidden_size, "output", init_bias=None)
    atom_hidden = tf.nn.relu(ainput + nei_message)

    fp = node_mask * atom_hidden
    fp = tf.reduce_sum(fp, 1)
    fp = tf.nn.relu(linearND(fp, hidden_size, "pooling"))

score = tf.squeeze(linear(fp, 1, "score"), [1])
loss = tf.nn.l2_loss(score - label) * 2 

tf.global_variables_initializer().run(session=session)

def load_data(path):
    data = []
    with open(path) as f:
        f.readline()
        for line in f:
            r,v = line.strip("\r\n ").split()
            data.append((r,float(v)))
    return data

def evaluate(data):
    sum_err = 0.0
    for it in xrange(0, len(data), batch_size):
        batch = data[it:it+batch_size]
        if len(batch) < batch_size:
            batch.extend(data[0:batch_size - len(batch)])
        src_batch, label_batch = zip(*batch)
        src_tuple = smiles2graph_list(src_batch)

        feed_map = {x:y for x,y in zip(src_holder, src_tuple)}
        feed_map.update({label:label_batch})
        err = session.run(loss, feed_dict=feed_map)
        sum_err += err
    return math.sqrt(sum_err / len(data))

saver = tf.train.Saver()
saver.restore(session, tf.train.latest_checkpoint(opts.model_path))
test = load_data(opts.test_path)
print "Test RMSE: %.4f" % evaluate(test)

