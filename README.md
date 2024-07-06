
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


### Procesamiento de los datasets:
los datasets obtenidos en formato txt tabulados fueron formateados a formato edgelist y nodelist con el siguiente codigo de python, ejecutado en kaggle, como paso previo se creo un dataset de kaggle en la ruta /kaggle/input con los archivos a formatear.Luego, cada dataset se formateo por separado reemplazando "dataset_name" en el codgio por el nombre de archivo correspondiente:

```

import pandas as pd
import os

# Ruta al archivo de entrada en Kaggle  (cambiar la ruta por la que corresponda al dataset a ejecutar)
input_file_path = '/kaggle/input/dataset_TP/dataset_name.txt'

# Ruta de salida en Kaggle/working
output_dir = '/kaggle/working/'

# Archivos de salida
edgelist_file = os.path.join(output_dir, 'dataset_name.edgelist')
nodelist_file = os.path.join(output_dir, 'dataset_namenodelist.txt')

# Leer el dataset completo
df = pd.read_csv(input_file_path, sep=" ")
#Nota:para datasets txt tabulados con espacios no tabulados (dataset de miRNAs), se utilizo: df = pd.read_csv(input_file_path, sep="\t")

# Obtener lista única de nodos (genes) y crear un DataFrame con índice
nodes = pd.concat([df['protein1'], df['protein2']]).unique()
node_df = pd.DataFrame(nodes, columns=['STRING_id'])
node_df.reset_index(inplace=True)
node_df.rename(columns={'index': 'index'}, inplace=True)

# Crear un diccionario para mapear STRING_id a índice
node_to_index = dict(zip(node_df['STRING_id'], node_df['index']))

# Crear el archivo nodelist.txt en el formato esperado
node_df.to_csv(nodelist_file, sep="\t", index=False)

# Reemplazar STRING_id con índice en el DataFrame original
df['protein1'] = df['protein1'].map(node_to_index)
df['protein2'] = df['protein2'].map(node_to_index)

# Ordenar las interacciones 
df = df.sort_values(by=['protein1', 'protein2'])

# Crear el archivo edgelist.edgelist en formato estándar
df[['protein1', 'protein2']].to_csv(edgelist_file, sep=" ", index=False, header=False)

print(f"Archivos '{edgelist_file}' y '{nodelist_file}' creados con éxito.")

```
En el codigo anterior, protein1 y protein2 son las columnas del dataset a considerar, en este caso las columnas de los datasets de interaccion de proteinas. Para otros datasets, sereemplaza por el nombre correspondiente.

luego, los archivos edgelist y nodelist obtenidos fueron incorporados a la carpeta de data del repositorio.



## 3. instrucciones de instalacion del codigo Bionev en kaggle y ejecucion de metodos de embeddings y Link prediction:

#### Installation

Para instalar directamente desde github a kaggle:

```bash
!git clone https://github.com/andresgfin/BioNEV.git
!pip install tensorflow
!pip install -e /kaggle/working/BioNEV
```

### Uso: para ejecutar los metodos de embeddings y tarea link prediccion se utilizo el siguiente comando, 
ejecutandolo a continuacion de la instalacion previa, en una celda de kaggle:

```
!python /kaggle/working/BioNEV/src/bionev/main.py --input /kaggle/working/BioNEV/data/folder_name/dataset_name.edgelist --method method_name --task link-prediction --output /kaggle/working/method_name_embeddings.txt

```
### ejemplo de uso, en este caso para ejecutar un dataset de miRNA con el metodo node2vec y link prediction:

```
!python /kaggle/working/BioNEV/src/bionev/main.py --input /kaggle/working/BioNEV/data/miRNA/mirna_enf.edgelist --method node2vec --task link-prediction --output /kaggle/working/node2vec.txt

```
Luego se repitio ejecutando cada dataset con cada metodo de embeddings con link prediction, reemplazando los nombres correspondientes en el comando anterior.




