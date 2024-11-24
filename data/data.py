import pandas as pd
from tqdm import tqdm
import pickle
import torch
import pickle
import numpy as np
import os
import math
import argparse
import json
def phrase_node2img_patch():
    df = pd.read_csv(f'data/{city}/node2img_patch.csv')

    return pd.Series(df.img_name.values,index=df.n_id).to_dict()


def get_node_index(city):
    node_index={}
    file=f'data/{city}/final_nodes.csv'
    data=pd.read_csv(file)
    node_index={data.loc[i,'id']:i for i in range(len(data))}
    index_node={i:data.loc[i,'id'] for i in range(len(data))}
    return node_index,index_node


def generate_pr(input,output,od=None):
    TopK=8
    node_features_list=[]
    with open(f'data/{city}/node_embedding_64.csv', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            features = [float(item) for item in line.split(',')[1:]]
            node_features_list.append(torch.Tensor(features))
    node_embedding = torch.stack(node_features_list)
    
    df = pd.read_csv(input)
    pathso = df['path'].values
    paths=[]
    lengths=[]
    max_node_length=0
    this_od = []
    for pathArr in tqdm(pathso):
        pathArr = json.loads(pathArr.replace('\'','\"'))[:TopK]
        o = pathArr[0]
        d = pathArr[-1]
        if od is not None and (o,d) in od:
            continue
        if len(pathArr)<TopK:
            continue
        for p in pathArr:
            max_node_length = max(max_node_length,len(p['path']))
            paths.append(p['path'])
            lengths.append(p['jaccard'])
            this_od.append((o,d))
            this_od.append((d,o))


    time_tensor = torch.tensor(lengths)

    node_features_list=[]
    with open(f'data/{city}/node_embedding_64.csv', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            features = [float(item) for item in line.split(',')[1:]]
            node_features_list.append(torch.Tensor(features))
    node_embedding = torch.stack(node_features_list)
    
    
    trip_time = paths
    times = lengths
    
    id2PatchName = phrase_node2img_patch()

    emb = pickle.load(open(f'data/{city}/image/resnet50.pkl','rb'))
    features_list=[]
    imageName2Number={}
    for index,name in enumerate(list(emb.keys())) :
        imageName2Number[name]=index
        features_list.append(torch.FloatTensor(emb[name]))
    
    if city == 'Aalborg':
        max_node_length = 132
        max_image_length = 563
        max_edge_length = 1319

    elif city=='Xian':
        max_node_length = 120
        max_image_length = 257
        max_edge_length = 920


    maxpath_nodes = 0
    maximage_nodes=0
    maxpath2Image_nodes=0
    maxedges=0

    path = np.ones((len(trip_time), max_node_length), dtype=int)*-4
    image = np.ones((len(trip_time),max_image_length), dtype=int)*-4
    graph_edge = np.zeros((len(trip_time),max_edge_length,2),dtype=int)
    path2Image = np.ones((len(trip_time), max_node_length), dtype=int)*-1
    times = torch.tensor(times)
    
    leng=0
    length_node = 0
    
    node_index,index_node=get_node_index(city)
    # df =pd.read_csv(f'data/{city}/nodes.csv')
    oids = index_node

    for index, i in enumerate(tqdm(trip_time)):
        pos_nodes = i
        last = -1
        path_nodes=[]
        path_nodes.append(-2)
        image_nodes=[]
        path2Image_nodes=[]
        edges=[]
        image_sep_index = []
        path_sep_index = []
        for node in pos_nodes:
            if last == -1:
                last = node
            else:
                last_id = oids[last]
                now_id = oids[node]
                last_patch = id2PatchName[last_id]
                now_patch = id2PatchName[now_id]
                last_image = "_".join(last_patch.split('_')[:-1])
                now_image = "_".join(now_patch.split('_')[:-1])
                last = node
                if now_image !=last_image:
                    path_sep_index.append(len(path_nodes))
                    path_nodes.append(-1)
                    
            path_nodes.append(node)
        
        length_node = max(length_node,len(path_nodes))

        num_error = 0
        for i,node in enumerate(path_nodes):
            if node == -2 or node == -1:
                image_nodes.append(node)
                image_sep_index.append(len(image_nodes)-1)
                path2Image_nodes.append(len(image_nodes)-1)
                imageName ="_".join(id2PatchName[oids[path_nodes[i+1]]].split('_')[:-1])
                for k in range(1,17):
                        try:
                            image_nodes.append(imageName2Number[imageName+f"_{k}.png"])
                        except Exception as e:
                            num_error=num_error+1
                            print(num_error,imageName+f"-{k}.png")
                            
                
            else:
                index0 = int(id2PatchName[oids[node]].split('_')[-1].split('.')[0])
                path2Image_nodes.append(len(image_nodes)-1-splitK * splitK+index0)
                
        # node节点的边
        for path_index,node in enumerate(path_nodes):
            image_index = path2Image_nodes[path_index]              
            last_image_sep = 0
            if node == -1:
                last_image_sep = path2Image_nodes[path_index]
            elif node != -2 and node != -1:
                # image_index = path2Image_nodes[path_index]

                if path_nodes[path_index-1] == -2:
                    path_before = len(path_nodes)-1
                elif path_nodes[path_index-1] == -1:
                    path_before = path_index-2
                else:
                    path_before = path_index-1
                
                if path_index+1 == len(path_nodes):
                    path_after = 1
                elif path_nodes[path_index+1] == -1:
                    path_after = path_index+2
                else:
                    path_after = path_index+1

                
                if image_nodes[image_index-1] == -2:
                    image_before = len(image_nodes)-1
                elif image_nodes[image_index-1] == -1:
                    image_before = image_index-2
                else:
                    image_before = image_index-1
                
                if image_index+1 == len(image_nodes):
                    image_after = 1
                elif image_nodes[image_index+1] == -1:
                    image_after = image_index+2
                else:
                    image_after = image_index+1

                # path - path本身
                edges.append((path_index,path_index))
                # image - image
                edges.append((image_index + max_node_length,image_index + max_node_length))
                # path - image
                edges.append((path_index,image_index + max_node_length))

                # path - path前一个
                edges.append((path_index, path_before))
                # path - path后一个
                edges.append((path_index, path_after))
                # path - path后一个对应的image
                edges.append((path_index, path2Image_nodes[path_after ] + max_node_length))
                # path - path前一个对应的image
                edges.append((path_index, path2Image_nodes[path_before] + max_node_length))

                
                # image - path前一个
                edges.append((image_index + max_node_length, path_before))
                # image - path后一个 
                edges.append((image_index + max_node_length, path_after))
                # image - path前一个对应的image 
                edges.append((image_index + max_node_length, path2Image_nodes[path_before] + max_node_length))
                # image - path后一个对应的image 
                edges.append((image_index + max_node_length, path2Image_nodes[path_after ] + max_node_length))


                col = ((image_index - last_image_sep)-1) %4 +1
                row = math.ceil((image_index - last_image_sep)/4.0)

                # if 'around' in mode:
                if col != 1:
                    # image前一个 - image
                    edges.append((image_before + max_node_length, image_index + max_node_length))
                    # image前一个 - path
                    edges.append((image_before + max_node_length, path_index))
                
                if col !=4:
                    # image后一个 - image
                    edges.append((image_after + max_node_length, image_index + max_node_length))
                    # image后一个 - path
                    edges.append((image_after + max_node_length, path_index))

                image_above = image_index - 4
                image_below = image_index + 4
                # 如果不是第一行
                if row != 1:
                    #image上一个 - image
                    edges.append((image_above + max_node_length, image_index + max_node_length))
                    #image上一个 - path
                    edges.append((image_above + max_node_length, path_index))

                if row != 4:
                    #image下一个 - image
                    edges.append((image_below + max_node_length, image_index + max_node_length))
                    #image下一个 - path
                    edges.append((image_below + max_node_length, path_index))

        
        edges.append((0,0))
        edges.append((0,max_node_length))
        edges.append((max_node_length,max_node_length))

        # 标记节点的边)
        for curr in range(len(path_sep_index)):
            
            last = curr - 1

            # path[sep] - image[sep]
            edges.append((path_sep_index[curr], path2Image_nodes[path_sep_index[curr]] + max_node_length))
            edges.append((path_sep_index[curr],path_sep_index[curr]))
            edges.append((path2Image_nodes[path_sep_index[curr]] + max_node_length, path2Image_nodes[path_sep_index[curr]] + max_node_length))
            # if 'mark' in mode:
            if 1==1:
                # path[sep] - path[cls]
                edges.append((path_sep_index[curr],0))
                # path[sep] - image[cls]
                edges.append((path_sep_index[curr], max_node_length))
                # path[sep] - 上一个path[sep]
                edges.append((path_sep_index[curr], path_sep_index[last]))
                # path[sep] - 上一个image[sep]
                edges.append((path_sep_index[curr], path2Image_nodes[path_sep_index[last]] + max_node_length))

                # image[sep] - path[cls]
                edges.append((path2Image_nodes[path_sep_index[curr]] + max_node_length, 0))
                # image[sep] - image[cls]
                edges.append((path2Image_nodes[path_sep_index[curr]] + max_node_length, max_node_length))
                # image[sep] - 上一个image[sep]
                edges.append((path2Image_nodes[path_sep_index[curr]] + max_node_length, path2Image_nodes[path_sep_index[last]] + max_node_length))
                # image[sep] - 上一个path[sep]
                edges.append((path2Image_nodes[path_sep_index[curr]] + max_node_length, path_sep_index[last]))

        edges = [(min(edge),max(edge)) for edge in edges]
        edges = list(set(edges))
        edges = [[edge[0],edge[1]] for edge in edges]
        

        path[index,0:len(path_nodes)] = path_nodes
        image[index,0:len(image_nodes)] = image_nodes
        path2Image[index,0:len(path2Image_nodes)] = path2Image_nodes
        graph_edge[index,0:len(edges),:] = edges

    #     maxpath_nodes = max(maxpath_nodes,len(path_nodes))
    #     maximage_nodes=max(maximage_nodes,len(image_nodes))
    #     maxpath2Image_nodes=max(maxpath2Image_nodes,len(path2Image_nodes))
    #     maxedges=max(maxedges,len(edges))

    # print(maxpath_nodes,maximage_nodes,maxpath2Image_nodes,maxedges)

    path = path+4
    image = image +4
    # print(leng)
    # print(length_node)
    # print(path.shape)
    directory = f'data/{city}/{opt.task}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file = open(f'{directory}/{output}.pkl', 'wb')

    path = path.reshape(-1,TopK,path.shape[1])
    image = image.reshape(-1,TopK,image.shape[1])
    path2Image = path2Image.reshape(-1,TopK,path2Image.shape[1])
    graph_edge = graph_edge.reshape(-1,TopK,graph_edge.shape[1],graph_edge.shape[2])
    time_tensor = time_tensor.reshape(-1,TopK)
    pickle.dump({'path':path,'image':image,'path2Image':path2Image,'image_features':torch.stack(features_list),'path_features':node_embedding,'times':time_tensor,'graph_edges':graph_edge}, file)
    file.close()

    return this_od

def generate(input,output,city,splitK):
    mode='mark+around'
    # splitK = 10
    node_features_list=[]
    with open(f'data/{city}/node_embedding_64.csv', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            features = [float(item) for item in line.split(',')[1:]]
            node_features_list.append(torch.Tensor(features))
    node_embedding = torch.stack(node_features_list)
    
    df = pd.read_csv(input)
    trip_time = df['path'].values
    times = df['time'].values
    
    id2PatchName = phrase_node2img_patch()
    emb = pickle.load(open(f'data/{city}/image/resnet50.pkl','rb'))
    features_list=[]
    imageName2Number={}
    for index,name in enumerate(list(emb.keys())) :
        name1 = name.replace('-',"_")
        imageName2Number[name1]=index
        features_list.append(torch.FloatTensor(emb[name]))
    
    if city == 'Aalborg':
        max_node_length = 134
        max_image_length = 750
        max_edge_length = 1156

    elif city=='Xian':
        max_node_length = 124
        max_image_length = 359
        max_edge_length = 1169

    path = np.ones((len(trip_time), max_node_length), dtype=int)*-4
    image = np.ones((len(trip_time),max_image_length), dtype=int)*-4
    graph_edge = np.zeros((len(trip_time),max_edge_length,2),dtype=int)
    path2Image = np.ones((len(trip_time), max_node_length), dtype=int)*-1
    times = torch.tensor(times)
    node_id2index,oids=get_node_index(city)
    
    leng=0
    length_node = 0
    


    max_path_nodes1,max_image_nodes1,max_path2Image1,max_graph_edge1=0,0,0,0

    for index, i in enumerate(tqdm(trip_time)):
        pos_nodes =  [node_id2index[int(k)] for k in i.split('-')]
        last = -1
        path_nodes=[]
        path_nodes.append(-2)
        image_nodes=[]
        path2Image_nodes=[]
        edges=[]
        image_sep_index = []
        path_sep_index = []
        for node in pos_nodes:
            if last == -1:
                last = node
            else:
                last_id = oids[last]
                now_id = oids[node]
                last_patch = id2PatchName[last_id]
                now_patch = id2PatchName[now_id]
                last_image = "_".join(last_patch.split('_')[:-1])
                now_image = "_".join(now_patch.split('_')[:-1])
                last = node
                if now_image !=last_image:
                    path_sep_index.append(len(path_nodes))
                    path_nodes.append(-1)
                    
            path_nodes.append(node)
        
        length_node = max(length_node,len(path_nodes))

        num_error = 0
        for i,node in enumerate(path_nodes):
            if node == -2 or node == -1:
                image_nodes.append(node)
                image_sep_index.append(len(image_nodes)-1)
                path2Image_nodes.append(len(image_nodes)-1)
                imageName ="_".join(id2PatchName[oids[path_nodes[i+1]]].split('_')[:-1])
                for k in range(1,splitK * splitK +1):
                        try:
                            image_nodes.append(imageName2Number[imageName+f"_{k}.png"])
                        except Exception as e:
                            num_error=num_error+1
                            print(num_error,imageName+f"_{k}.png")
                            
                
            else:
                index0 = int(id2PatchName[oids[node]].split('_')[-1].split('.')[0])
                path2Image_nodes.append(len(image_nodes)-1-splitK * splitK+index0)

        for path_index,node in enumerate(path_nodes):
            image_index = path2Image_nodes[path_index]              
            last_image_sep = 0
            if node == -1:
                last_image_sep = path2Image_nodes[path_index]
            elif node != -2 and node != -1:

                if path_nodes[path_index-1] == -2:
                    path_before = len(path_nodes)-1
                elif path_nodes[path_index-1] == -1:
                    path_before = path_index-2
                else:
                    path_before = path_index-1
                
                if path_index+1 == len(path_nodes):
                    path_after = 1
                elif path_nodes[path_index+1] == -1:
                    path_after = path_index+2
                else:
                    path_after = path_index+1

                
                if image_nodes[image_index-1] == -2:
                    image_before = len(image_nodes)-1
                elif image_nodes[image_index-1] == -1:
                    image_before = image_index-2
                else:
                    image_before = image_index-1
                
                if image_index+1 == len(image_nodes):
                    image_after = 1
                elif image_nodes[image_index+1] == -1:
                    image_after = image_index+2
                else:
                    image_after = image_index+1

                # path - path
                edges.append((path_index,path_index))
                # image - image
                edges.append((image_index + max_node_length,image_index + max_node_length))
                # path - image
                edges.append((path_index,image_index + max_node_length))

                # path - path previous
                edges.append((path_index, path_before))
                # path - path next
                edges.append((path_index, path_after))
                # path - path next image
                edges.append((path_index, path2Image_nodes[path_after ] + max_node_length))
                # path - path previous image
                edges.append((path_index, path2Image_nodes[path_before] + max_node_length))

                
                # image - path previous
                edges.append((image_index + max_node_length, path_before))
                # image - path next 
                edges.append((image_index + max_node_length, path_after))
                # image - path previous image 
                edges.append((image_index + max_node_length, path2Image_nodes[path_before] + max_node_length))
                # image - path next image 
                edges.append((image_index + max_node_length, path2Image_nodes[path_after ] + max_node_length))


                col = ((image_index - last_image_sep)-1) %splitK +1
                row = math.ceil((image_index - last_image_sep)/splitK)

                if 'around' in mode:
                    if col != 1:
                        # imageprevious - image
                        edges.append((image_before + max_node_length, image_index + max_node_length))
                        # image previous - path
                        edges.append((image_before + max_node_length, path_index))
                    
                    if col !=splitK:
                        # image next - image
                        edges.append((image_after + max_node_length, image_index + max_node_length))
                        # image next - path
                        edges.append((image_after + max_node_length, path_index))

                    image_above = image_index - splitK
                    image_below = image_index + splitK

                    if row != 1:
                        #image above  - image
                        edges.append((image_above + max_node_length, image_index + max_node_length))
                        #image above  - path
                        edges.append((image_above + max_node_length, path_index))

                    if row != splitK:
                        #image below  - image
                        edges.append((image_below + max_node_length, image_index + max_node_length))
                        #image below  - path
                        edges.append((image_below + max_node_length, path_index))

        
        edges.append((0,0))
        edges.append((0,max_node_length))
        edges.append((max_node_length,max_node_length))

        for curr in range(len(path_sep_index)):
            
            last = curr - 1

            # path[sep] - image[sep]
            edges.append((path_sep_index[curr], path2Image_nodes[path_sep_index[curr]] + max_node_length))
            edges.append((path_sep_index[curr],path_sep_index[curr]))
            edges.append((path2Image_nodes[path_sep_index[curr]] + max_node_length, path2Image_nodes[path_sep_index[curr]] + max_node_length))
            if 'mark' in mode:
                # path[sep] - path[cls]
                edges.append((path_sep_index[curr],0))
                # path[sep] - image[cls]
                edges.append((path_sep_index[curr], max_node_length))
                # path[sep] - previous path[sep]
                edges.append((path_sep_index[curr], path_sep_index[last]))
                # path[sep] - previous image[sep]
                edges.append((path_sep_index[curr], path2Image_nodes[path_sep_index[last]] + max_node_length))

                # image[sep] - path[cls]
                edges.append((path2Image_nodes[path_sep_index[curr]] + max_node_length, 0))
                # image[sep] - image[cls]
                edges.append((path2Image_nodes[path_sep_index[curr]] + max_node_length, max_node_length))
                # image[sep] - previous image[sep]
                edges.append((path2Image_nodes[path_sep_index[curr]] + max_node_length, path2Image_nodes[path_sep_index[last]] + max_node_length))
                # image[sep] - previous path[sep]
                edges.append((path2Image_nodes[path_sep_index[curr]] + max_node_length, path_sep_index[last]))

        edges = [(min(edge),max(edge)) for edge in edges]
        edges = list(set(edges))
        edges = [[edge[0],edge[1]] for edge in edges]
        
        path[index,0:len(path_nodes)] = path_nodes
        image[index,0:len(image_nodes)] = image_nodes
        path2Image[index,0:len(path2Image_nodes)] = path2Image_nodes
        graph_edge[index,0:len(edges),:] = edges

    path = path+4
    image = image +4
    directory = f'data/{city}/{opt.task}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file = open(f'{directory}/{output}.pkl', 'wb')
    pickle.dump({'path':path,'image':image,'path2Image':path2Image,'image_features':torch.stack(features_list),'path_features':node_embedding,'times':times,'graph_edges':graph_edge}, file)
    file.close()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_num',default=4, type=int)
    parser.add_argument('--city',default='Xian', type=str)
    parser.add_argument('--task',default='path_ranking', type=str)
    opt, unknown = parser.parse_known_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    city = opt.city
    splitK = opt.split_num
    if opt.task == 'pretrain':
        generate(f'data/{city}/{opt.task}/pretrain.csv','pretrain',city,splitK)
    
    elif opt.task=='travel_time_estimation':
        datasets = ['finetune','valid','test']
        for dataset in datasets:
            generate(f'data/{city}/{opt.task}/{dataset}.csv',dataset,city,splitK)
    
    elif opt.task=='path_ranking':
        datasets = ['valid','test']
        dataset = 'finetune'
        ft_od = generate_pr(f'data/{city}/{opt.task}/{dataset}.csv',dataset)
        for dataset in datasets:
            ft_od = generate_pr(f'data/{city}/{opt.task}/{dataset}.csv',dataset,ft_od)