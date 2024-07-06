
# BioNEV (Biomedical Network Embedding Evaluation)

## 1. Este repositorio contiene codigo python adaptado de:

```
@article{yue2020graph,
  title={Graph embedding on biomedical networks: methods, applications and evaluations},
  author={Yue, Xiang and Wang, Zhen and Huang, Jingong and Parthasarathy, Srinivasan and Moosavinasab, Soheil and Huang, Yungui and Lin, Simon M and Zhang, Wen and Zhang, Ping and Sun, Huan},
  journal={Bioinformatics},
  volume={36},
  number={4},
  pages={1241--1251},
  year={2020},
  publisher={Oxford University Press}
}
```

## 2. instrucciones de uso:
###Datasets y metodos de embeddings:

Este repositorio contiene 9 datasets (8 datasets de interacciones proteina-proteina de disferentes especies, y 2 datasets de miRNAs), en formato edgelist en la carpeta "data". Los datasets de interraccion de proteinas son: 
C.elegans, Acanthomoeba, E.coli, Oceanococcus, Oceanobacter, Jonesia denitrificans, Spirulina
los datasets de miRNAs son: miRNA_gen (miRNAs y sus genes target) y miRNA_enf (miRNAs asociados a enfermedades)

Dichos datasets se utilizaron para evaluar 10 metodos de mebeddings representativos, con la tarea link prediction.

los metodos de embeddings evaluados son:
- matrix factorization-based: Laplacian Eigenmap, SVD, Graph Factorization, HOPE, GraRep
- random walk-based: DeepWalk, node2vec, struc2vec
- neural network-based: LINE, SDNE


### codigo para procesar los datasets:
los datasets obtenidos en formato txt tabulados fueron formateados a formato edgelist y nodelist con el siguiente codigo de python, ejecutado en kaggle:


import pandas as pd
import os

# Ruta al archivo de entrada en Kaggle  (cambiar la ruta por la que corresponda al dataset a ejecutar)
input_file_path = '/kaggle/input/dataset_TP/tabla_miRNA_gen.txt'

# Ruta de salida en Kaggle/working
output_dir = '/kaggle/working/'

# Archivos de salida
edgelist_file = os.path.join(output_dir, 'mirna.edgelist')
nodelist_file = os.path.join(output_dir, 'nodelist.txt')

# Leer el dataset completo
#para tabla txt tabulados con espacios usar: df = pd.read_csv(input_file_path, sep="\t")
df = pd.read_csv(input_file_path, sep="\t")

# Obtener lista única de nodos (genes) y crear un DataFrame con índice
nodes = pd.concat([df['miRNA'], df['Validated target']]).unique()
node_df = pd.DataFrame(nodes, columns=['STRING_id'])
node_df.reset_index(inplace=True)
node_df.rename(columns={'index': 'index'}, inplace=True)

# Crear un diccionario para mapear STRING_id a índice
node_to_index = dict(zip(node_df['STRING_id'], node_df['index']))

# Crear el archivo nodelist.txt en el formato esperado
node_df.to_csv(nodelist_file, sep="\t", index=False)

# Reemplazar STRING_id con índice en el DataFrame original
df['miRNA'] = df['miRNA'].map(node_to_index)
df['Validated target'] = df['Validated target'].map(node_to_index)

# Ordenar las interacciones 
df = df.sort_values(by=['miRNA', 'Validated target'])

# Crear el archivo edgelist.edgelist en formato estándar
df[['miRNA', 'Validated target']].to_csv(edgelist_file, sep=" ", index=False, header=False)

print(f"Archivos '{edgelist_file}' y '{nodelist_file}' creados con éxito.")





## 3. Code
The graph embedding learning for Laplician Eigenmap, Graph Factorization, HOPE, GraRep, DeepWalk, node2vec, LINE, SDNE uses the code from [OpenNE](https://github.com/thunlp/OpenNE)
The code of [struc2vec](https://github.com/leoribeiro/struc2vec) and [GAE](https://github.com/tkipf/gae) is from their authors. 
To ensure different source code could run successfully in our framework, we modify part of their source code.
 
#### Installation

Use the following command to install directly from GitHub;

```bash
$ pip install git+https://github.com/xiangyue9607/BioNEV.git
```



#### General Options
- --input, input graph file. Only accepted edgelist format. 
- --output, output graph embedding file. 
- --task, choose to evaluate the embedding quality based on a specific prediction task (i.e., link-prediction, node-classification, none (no eval), default is none) 
- --testing-ratio, testing set ratio for prediction tasks. Only applied when --task is not none. The default is 0.2 
- --dimensions, the dimensions of embedding for each node. The default is 100. 
- --method, the name of embedding method 
- --label-file, the label file for node classification.  
- --weighted, true if the input graph is weighted. The default is False.
- --eval-result-file, the filename of eval result (save the evaluation result into a file). Skip it if there is no need.
- --seed, random seed. The default is 0. 

#### Specific Options

- Matrix Factorization-based methods:
  - --kstep, k-step transition probability matrix for GraRep. The default is 4. It must divide the --dimension.
  - --weight-decay, coefficient for L2 regularization for Graph Factorization. The default is 5e-4.
  - --lr, learning rate for gradient descent in Graph Factorization. The default is 0.01.

- Random Walk-based methods:
  - --number-walks, the number of random walks to start at each node.
  - --walk-length, the length of the random walk started at each node.
  - --window-size, window size of node sequence. 
  - --p, --q, two parameters that control how fast the walk explores and leaves the neighborhood of starting node. The default values of p, q are 1.0.
  - --OPT1, --OPT2, --OPT3, three running time efficiency optimization strategies for struc2vec. The default values are True.
  - --until-layer, calculation until the layer. A hyper-parameter for struc2vec. The default is 6.
  
- Neural Network-based methods:
  - --lr, learning rate for gradient descent. The default is 0.01.
  - --epochs, training epochs. The default is 5. Suggest to set a small value for LINE and SDNE (e.g., 5), and a large value for GAE (e.g., 500).
  - --bs, batch size. Only applied for SDNE. The default is 200.
  - --negative-ratio, the negative sampling ratio for LINE. The default is 5.
  - --order, the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order. The default is 2.
  - --alpha, a hyperparameter in SDNE that balances the weight of 1st-order and 2nd-order proximities. The default is 0.3.
  - --beta', a hyperparameter in SDNE that controls the reconstruction weight of the nonzero elementsin the training graph. The default is 0.
  - --dropout, dropout rate. Only applied for GAE. The default is 0.
  - --hidden, number of units in hidden layer. Only applied for GAE. The default is 32.
  - --gae_model_selection, GAE model variants: gcn_ae or gcn_vae. The default is gcn_ae.

#### Running example

```
bionev --input ./data/DrugBank_DDI/DrugBank_DDI.edgelist \
       --output ./embeddings/DeepWalk_DrugBank_DDI.txt \
       --method DeepWalk \
       --task link-prediction \
       --eval-result-file eval_result.txt
```

```
bionev --input ./data/Clin_Term_COOC/Clin_Term_COOC.edgelist \
       --label-file ./data/Clin_Term_COOC/Clin_Term_COOC_labels.txt \
       --output ./embeddings/LINE_COOC.txt \
       --method LINE \
       --task node-classification \
       --weighted True
```


