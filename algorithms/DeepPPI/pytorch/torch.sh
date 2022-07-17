#!/bin/bash                                                                                                                                                                                                        # Nom du job                                                                                                                                                                                                       
#SBATCH --job-name=deepPPI%j                                                                                                                                                                                       
#SBATCH --export=ALL                                                                                                                                                                                               
# Nom de la partition                                                                                                                                                                                              
#SBATCH -p Piramid                                                                                                                                                                                                 
# Specification du noeud (budbud002 : P100, budbud001 : K80)                                                                                                                                                       
#SBATCH --nodelist=budbud002                                                                                                                                                                                       
# Nombre de CPUs par tache                                                                                                                                                                                         
#SBATCH --cpus-per-task=10                                                                                                                                                                                         
# Temps maximum du job (j:h:m)                                                                                                                                                                                     
#SBATCH -t 3-00:00:00                                                                                                                                                                                              
#SBATCH --output=slurm-%j.out                                                                                                                                                                                      
#SBATCH --nodes=1                                                                                                                                                                                                  
#SBATCH -e slurm-%j.err-%N                                                                                                                                                                                         #SBATCH --exclusive=mcs                                                                                                                                                                                            
#SBATCH --gres=gpu:P100:0                                                                                                                                                                                          
#Definition de repertoires                                                                                                                                                                                          
HOMEDIR=/home/UFIP/servantie_c/deepppi/PPIpredict/deep
DATADIR=/home/UFIP/servantie_c/deepppi/PPIpredict/deep/data
LOGS=/home/UFIP/servantie_c/deepppi/PPIpredict/deep/results/ccipl
VENVDIR=/home/UFIP/servantie_c/deepppi/ccipl/venv

#on nettoie l'environnement                                                                                                                                                                                         
module purge

echo 'Chargement du module CUDA'
#on charge cuda                                                                                                                                                                                                     
module load cuda/8.0.61
echo 'CUDA charge, chargement du module python'
#on charge python (la version qui a pytorch)                                                                                                                                                                        
module load python/3.6.5
echo 'preparing split files of 1M'
#python3 data/miniscript.py -f $DATADIR/split20                                                                                                                                                                     
ls
echo 'choosing to create a test file of a certain size'
python3 data/pick_data.py -x 10 -start 12 -f data/splits/split_test_10

echo 'creating a train file of a certain size'
python3 data/pick_data.py -x 10 -start 0 -f data/splits/split_train_10


echo 'Launching a run'
#python3 test_cci.py (test si cuda fonctionne et pytorch)                                                                                                                                                          #commande ayant des soucis de dossiers 
python3 main.py -lr 0.001 -epochs 200 -f slurmTest1 -data data/splits/split_train_10 -gpu 0 -b 2 -model 2 -o 2 -save "save"

echo 'All done, should be ok' >> $LOGS/log

