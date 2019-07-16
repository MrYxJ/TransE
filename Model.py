from  metric import  *

def TransE(folder):
   pass

def TestTransE():
    folder = 'data/0_1/'
    #TransE(folder)
    test = []
    with open(folder + 'ref_ent_ids') as f:
        for line in f.readlines():
            test.append((int(line.split('\t')[0].strip()),int(line.split('\t')[1].strip())))

    vec = np.load(folder + 'ent_embeddings.npy')
    get_hits(vec,test)

if __name__ == '__main__':
    TestTransE()

