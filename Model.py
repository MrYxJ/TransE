import math
import tensorflow as tf
from  metric import  *
import random
import os
import copy

from embed_utils import *
from triples_data import *

batch_size = 10000
embed_size = 200
num_epochs = 200
learning_rate = 0.01
alpha = 0
print_validation = 10
sim_loss_param = 0.05
inner_sim_param = 0.05
neg_param = 0.1
early_stop = 0.0005

l1 = 1.0
l2 = 2.0

valid_mul = 1

def only_pos_loss(phs, prs, pts):
    # base_loss = tf.reduce_sum(tf.reduce_sum(tf.abs(phs + prs - pts), 1))
    base_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(phs + prs - pts, 2), 1))   # 按行求和
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(base_loss)

    return optimizer , base_loss

def only_neg_loss(nhs, nrs, nts):
    neg_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(nhs + nrs - nts, 2), 1))
    base_loss = - neg_param * neg_loss
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(base_loss)
    return optimizer , base_loss

def generate_pos_batch_of2KBs(triples_data1, triples_data2, step):
    # print(triples_data1.train_triples[0: 2])
    # print(triples_data2.train_triples[0: 2])
    assert batch_size % 2 == 0
    num1 = int(triples_data1.train_triples_num / (
        triples_data1.train_triples_num + triples_data2.train_triples_num) * batch_size)
    num2 = batch_size - num1
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > triples_data1.train_triples_num:
        end1 = triples_data1.train_triples_num
    if end2 > triples_data2.train_triples_num:
        end2 = triples_data2.train_triples_num
    pos_triples1 = triples_data1.train_triples[start1: end1]
    pos_triples2 = triples_data2.train_triples[start2: end2]
    return pos_triples1, pos_triples2

def generate_neg_triples(pos_triples, triples_data, neg_scope):
    neg_triples = list()
    for (h, r, t) in pos_triples:
        h2, r2, t2 = h, r, t
        temp_scope, num = neg_scope, 0
        while True:
            choice = random.randint(0, 999)  # 这是一个随机生成负例的小trick把，0.5概率换h或者t
            if choice < 500:
                if temp_scope:
                    h2 = random.sample(triples_data.r_hs_train[r], 1)[0] #
                else:
                    h2 = random.sample(triples_data.ents_list, 1)[0]
                    # h2 = random.sample(triples_data.heads_list, 1)[0]
            elif choice >= 500:
                if temp_scope:
                    t2 = random.sample(triples_data.r_ts_train[r], 1)[0]
                else:
                    t2 = random.sample(triples_data.ents_list, 1)[0]
                    # t2 = random.sample(triples_data.tails_list, 1)[0]
            if not triples_data.exist(h2, r2, t2):
                break
            else:
                num += 1
                if num > 10:
                    temp_scope = False
        neg_triples.append((h2, r2, t2))
    return neg_triples

def generate_pos_neg_batch(triples_data1, triples_data2, step, is_half=False, neg_scope=False, multi=1):
    pos_triples1, pos_triples2 = generate_pos_batch_of2KBs(triples_data1, triples_data2, step)

    if is_half:
        pos_triples11 = random.sample(pos_triples1, len(pos_triples1) // 2)
        pos_triples22 = random.sample(pos_triples2, len(pos_triples2) // 2)
        neg_triples1 = generate_neg_triples(pos_triples11, triples_data1, neg_scope)
        neg_triples2 = generate_neg_triples(pos_triples22, triples_data2, neg_scope)
    else:
        neg_triples1 = generate_neg_triples(pos_triples1, triples_data1, neg_scope)
        neg_triples2 = generate_neg_triples(pos_triples2, triples_data2, neg_scope)

    neg_triples1.extend(neg_triples2)
    if multi > 1:
        for i in range(multi - 1):
            neg_triples1.extend(generate_neg_triples(pos_triples1, triples_data1, neg_scope))
            neg_triples1.extend(generate_neg_triples(pos_triples2, triples_data2, neg_scope))

    pos_triples1.extend(pos_triples2)
    return pos_triples1, neg_triples1

def generate_input(folder):
    triples1 = read_triples_ids(folder + 'triples_1')
    triples_data1 = Triples_Data(triples1)

    triples2 = read_triples_ids(folder + 'triples_2')
    triples_data2 = Triples_Data(triples2)

    ent_num = len(triples_data1.ents | triples_data2.ents)
    rel_num = len(triples_data1.rels | triples_data2.rels)
    triples_num = len(triples1) + len(triples2)
    print('all ents:', ent_num)
    print('all rels:', len(triples_data1.rels), len(triples_data2.rels), rel_num)
    print('all triples: %d + %d = %d' % (len(triples1), len(triples2), triples_num))

    refs1, refs2 = read_ref(folder + 'ref_ent_ids')
    ref1_list = copy.deepcopy(refs1)
    ref2_list = copy.deepcopy(refs2)

    print("To align:", len(refs2))
    sup_ents_pairs = read_pair_ids(folder + 'sup_ent_ids')

    return triples_data1, triples_data2, sup_ents_pairs, refs1, ref2_list, refs2, ref1_list, triples_num, ent_num, rel_num

def valid(ent_embeddings, references_s, references_t_list, references_t, references_s_list, early_stop_flag1,
          early_stop_flag2, hits, top_k=[1, 5, 10, 50, 100], valid_threads=valid_mul):
    # if valid_threads > 1:
    #     res1, hits1 = valid_results_mul(ent_embeddings, references_s, references_t_list, 'X-EN:', top_k=top_k)
    #     res2, hits2 = valid_results_mul(ent_embeddings, references_t, references_s_list, 'EN-X:', top_k=top_k)
    # else:
    res1, hits1 = valid_results(ent_embeddings, references_s, references_t_list, 'X-EN:', top_k=top_k)
    res2, hits2 = valid_results(ent_embeddings, references_t, references_s_list, 'EN-X:', top_k=top_k)
    flag1 = early_stop_flag1 - res1
    flag2 = early_stop_flag2 - res2
    flag3 = hits1 - hits
    if flag1 < early_stop and flag2 < early_stop and flag3 < 0:
        print("early stop")
        return -1, -1, -1
    else:
        return res1, res2, hits1


def valid_results(embeddings, references_s, references_t, word, top_k=[1, 5, 10, 50, 100]):
    s_len = int(references_s.shape[0])
    t_len = int(references_t.shape[0])
    s_embeddings = tf.nn.embedding_lookup(embeddings, references_s)
    t_embeddings = tf.nn.embedding_lookup(embeddings, references_t)
    similarity_mat = tf.matmul(s_embeddings, t_embeddings, transpose_b=True)
    t = time.time()
    sim = similarity_mat.eval()
    num = [0 for k in top_k]
    mean = 0
    for i in range(s_len):
        ref = i
        rank = (-sim[i, :]).argsort()
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    acc = np.array(num) / s_len
    for i in range(len(acc)):
        acc[i] = round(acc[i], 4)
    mean /= s_len
    print("{} acc of top {} = {}, mean = {:.3f}, time = {:.3f} s ".
          format(word, top_k, acc, mean, time.time() - t))
    return mean / t_len, acc[2]

def TransE(folder):
    triples_data1, triples_data2, sup_ents_pairs, ref_s, ref_t_list, ref_t, ref_s_list, triples_num, ent_num, rel_num = generate_input(folder)
    graph = tf.Graph()
    with graph.as_default():
        pos_hs = tf.placeholder(tf.int32, shape=[None])
        pos_rs = tf.placeholder(tf.int32, shape=[None])
        pos_ts = tf.placeholder(tf.int32, shape=[None])
        neg_hs = tf.placeholder(tf.int32, shape=[None])
        neg_rs = tf.placeholder(tf.int32, shape=[None])
        neg_ts = tf.placeholder(tf.int32, shape=[None])
        flag = tf.placeholder(tf.bool)

        with tf.variable_scope('relation2vec' + 'embedding'):
            ent_embeddings = tf.Variable(tf.truncated_normal([ent_num, embed_size], stddev=1.0 / math.sqrt(embed_size)))
            rel_embeddings = tf.Variable(tf.truncated_normal([rel_num, embed_size], stddev=1.0 / math.sqrt(embed_size)))
            ent_embeddings = tf.nn.l2_normalize(ent_embeddings, 1)
            rel_embeddings = tf.nn.l2_normalize(rel_embeddings, 1)
            references_s = tf.constant(ref_s, dtype=tf.int32)
            references_t_list = tf.constant(ref_t_list, dtype=tf.int32)
            references_t = tf.constant(ref_t, dtype=tf.int32)
            references_s_list = tf.constant(ref_s_list, dtype=tf.int32)

        phs = tf.nn.embedding_lookup(ent_embeddings, pos_hs)
        prs = tf.nn.embedding_lookup(rel_embeddings, pos_rs)
        pts = tf.nn.embedding_lookup(ent_embeddings, pos_ts)
        nhs = tf.nn.embedding_lookup(ent_embeddings, neg_hs)
        nrs = tf.nn.embedding_lookup(rel_embeddings, neg_rs)
        nts = tf.nn.embedding_lookup(ent_embeddings, neg_ts)
        optimizer, loss = tf.cond(flag, lambda: only_pos_loss(phs, prs, pts), lambda: only_neg_loss(nhs, nrs, nts))

        total_start_time = time.time()
        early_stop_flag1, early_stop_flag2, hits = 1, 1, 0

        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()
            num_steps = triples_num // batch_size
            for epoch in range(num_epochs):
                pos_loss = 0
                start = time.time()
                for step in range(num_steps):
                    batch_pos, batch_neg = generate_pos_neg_batch(triples_data1, triples_data2, step)
                    for i in range(2):
                        train_flag = True if i % 2 == 0 else False
                        feed_dict = {pos_hs: [x[0] for x in batch_pos],
                                     pos_rs: [x[1] for x in batch_pos],
                                     pos_ts: [x[2] for x in batch_pos],
                                     neg_hs: [x[0] for x in batch_neg],
                                     neg_rs: [x[1] for x in batch_neg],
                                     neg_ts: [x[2] for x in batch_neg],
                                     }
                        (_, loss_val) = sess.run([optimizer, loss], feed_dict=feed_dict)
                        pos_loss += loss_val
                random.shuffle(triples_data1.train_triples)
                random.shuffle(triples_data2.train_triples)
                end = time.time()
                print("{}/{}, relation_loss = {:.3f}, time = {:.3f} s".format(epoch, num_epochs, pos_loss, end - start))
                # if (epoch % print_validation == 0 or epoch == num_epochs - 1) and epoch >= 200:
                if epoch % print_validation == 0 or epoch == num_epochs - 1:
                    early_stop_flag1, early_stop_flag2, hits = valid(ent_embeddings, references_s, references_t_list,
                                                                     references_t, references_s_list, early_stop_flag1,
                                                                     early_stop_flag2, hits)
                    if early_stop_flag1 < 0 and early_stop_flag2 < 0 and hits < 0: # 如果Hits结果降低就提前退出
                        print(time.time() - total_start_time)
                        np.save(folder + 'jape_ent_embeddings', sess.run(ent_embeddings))
                        exit()

            np.save(folder + 'ent_embeddings', sess.run(ent_embeddings))

def TestTransE():
    folder = 'data/0_3/'
    if  os.path.exists(folder + 'ent_embeddings.npy'):
        TransE(folder)

    c = input(folder+ ': ')
    test = []
    with open(folder + 'ref_ent_ids') as f:
        for line in f.readlines():
            test.append((int(line.split('\t')[0].strip()),int(line.split('\t')[1].strip())))

    vec = np.load(folder + 'ent_embeddings.npy')
    print(vec[0])
    get_hits(vec,test)

if __name__ == '__main__':
    TestTransE()

