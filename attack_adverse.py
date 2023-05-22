import pandas as pd 
import tensorflow as tf 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
tf.random.set_seed(42) 
from alive_progress import alive_bar
import time
from keras import backend as K
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
scaler = MinMaxScaler()
categorical_cross_entropy = tf.keras.losses.CategoricalCrossentropy()



attacker_dataset= pd.read_csv('attacker_dataset.csv')
attacker_dataset=attacker_dataset.drop(columns = ['attack'])
X_train_attacker, X_test_attacker, y_train_attacker, y_test_attacker = train_test_split(attacker_dataset.drop(columns = ["category"]), attacker_dataset.category, stratify=attacker_dataset.category, shuffle=True, test_size=0.25,random_state=42)
test_substitute = pd.concat([X_test_attacker, y_test_attacker], axis=1)

# Model attacker Loading
attacker_model = tf.keras.models.load_model('./models_attacker/dnnattacker', custom_objects={'categorical_cross_entropy': categorical_cross_entropy})
#attacker_model = pkl.load(open( './models_attacker/rfattacker.pkl', 'rb'))

# Model defender Loading
defender_model= tf.keras.models.load_model('./models_defender/dnndefender', custom_objects={'categorical_cross_entropy': categorical_cross_entropy})
#defender_model= pkl.load(open('./models_defender/rfdefender.pkl', 'rb'))


dataset_input = test_substitute # From the attacker
model_input = attacker_model
# The mask combinaisons that can be adapted in function of the studied domain. Depends of the manipulable factors
combinaisons = [ # Don't take [000] because not relevant, so 7 combinaisons. [001] = duration, [010] = totpkt et [100] sbytes 
    [0,0,1],
    [0,1,0],
    [1,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0],
    [1,1,1]
]

# Define max value of each modified feature in the general dataset to project too big values on these max
# It's the max value of the attacker or defender dataset, specified in parameter
max_dur = dataset_input['dur'].max()
max_pkts = dataset_input['spkts'].max()
max_out = dataset_input['sbytes'].max()

# Used to generate Adv Ex. Don't take the label to generate adv ex
ben_dataset = dataset_input.loc[dataset_input['category'] == 0]
mal_dataset = dataset_input.loc[dataset_input['category'] == 1] # 1,2, ......

# Mean determination to determine during peturbation generation the direction of the perturbation (negative or positive)
ben_mean_dur = ben_dataset['dur'].mean()
ben_mean_Spkts = ben_dataset['spkts'].mean()
ben_mean_Sbytes = ben_dataset['sbytes'].mean()

# Reduce the dataset for the tests to speed up the generation. Just take 1000 instances here
#mal_dataset_reduced, not_used = train_test_split(mal_dataset, shuffle=True, train_size=(1000/mal_dataset.shape[0])) # , random_state=42
mal_dataset_reduced=mal_dataset
mal_dataset_reduced=mal_dataset_reduced.drop(columns = ['category'])
scaler.fit(mal_dataset_reduced.to_numpy()) # to numpy to avoid the warning later when we predict with a numpy instead of dataframe

adv_ex = []
total_ex = []
nb_of_needed_step = 0

tot_masks = []# Used to know what the most used mask to create an adversarial example
index_of_mask = 0

#max_ratio = dataset_input['RatioOutIn'].max() # Max value in RatioOutIn for the semantic constraints
start = time.process_time()
with alive_bar(len(mal_dataset_reduced)) as bar:
    # For each malicious instance
    for index, row in mal_dataset_reduced.iterrows():
        breaked = False
        perturb_direction = []
        # Check the direction of perturbation for the 4 instance features
        if(row[19] <= ben_mean_Sbytes): # Out
            perturb_direction.append(1)
        else:
            perturb_direction.append(-1)          
        if(row[17] <= ben_mean_Spkts): # spkts
            perturb_direction.append(1)
        else:
            perturb_direction.append(-1)              
        if(row[14] <= ben_mean_dur): # dur
            perturb_direction.append(1)
        else:
            perturb_direction.append(-1)
        dif_mean_dur = ben_dataset[['dur']].mean() - row[14]
        dif_mean_dur = abs(dif_mean_dur[0])
        dif_mean_pkts = ben_dataset[['spkts']].mean() -row[17]
        dif_mean_pkts = abs(dif_mean_pkts[0])
        dif_mean_out = ben_dataset[['sbytes']].mean() - row[19]
        dif_mean_out = abs(dif_mean_out[0])
        

        # Max 10 iterations of iterative perturbation to try to get benign instance
        for i in range(1, 10):
            nb_of_needed_step += 1 # start directly at the round 1
            # Iterate while not benign 
            if(breaked==False):
                # For each 7 combinations of perturbations
                for combi in combinaisons:
                    index_of_mask += 1 # check which mask is used
                    # add perturbation to the autorized features                    
                    adv = np.array(row)

                    perturb1 = np.array(combi[0]) * ( dif_mean_out * (i*0.001) * perturb_direction[0])
                    perturb2 = np.array(combi[1]) * (dif_mean_pkts * (i*0.001) * perturb_direction[1])
                    perturb3 = np.array(combi[2]) * (dif_mean_dur * (i*0.001) * perturb_direction[2])

                    # Addition of crafted perturbation
                    adv[19] = adv[19] + perturb1 # sBytes
                    adv[17] = adv[17] + perturb2 # sPackets
                    adv[17] = int(adv[17] ) 
                    adv[14] = adv[14] + perturb3 # duration  # cast in INT to keep only the integer value

                    # Syntactic Constraints
                    # Add projection on the max value present in the dataset to keep the physical limitation
                    if(adv[14] > max_dur):
                        adv[14] = max_dur
                    if(adv[19] > max_out):
                        adv[19] = max_out
                    if(adv[17] > max_pkts):
                        adv[17] = max_pkts

                    # if there is new bytes, normaly there is also at least 1 packet
                    if(adv[17] == 0 and adv[19] > 0):
                        adv[17] = 1 # Maybe change this part

                    if(adv[19]/adv[17]>1500):
                        n=int((adv[19]-row[19])/1500)
                        adv[17]=adv[17]+n+1          
                    # Add the Semantic Contraints
                    # Total number of Bytes in the communication. Sum of sbytes and InBytes feature values.
                    adv[16] = adv[19]+adv[20] # TotBytes
                    adv[15] = adv[17]+adv[18] # Totpakts

                    # Average number of bytes exchanged per packet. Ratio between TotBytes and spkts.
                    adv[21] = adv[15]/adv[14] # raet
                    # Average number of bytes exchanged per second. Ratio between TotBytes and duration.
                    adv[22] = adv[17]/adv[14] # Sraet
                    # Average number of packets exchanged per second. Ratio between spkts and duration.
                    adv[23] = adv[18]/adv[14] # Draet

                    adv2 = [] # used to fit with the input of the model because normaly take a matrix, so need the matrix notation, even for a vector
                    adv2.append(adv)
                    adv2_scaled = scaler.transform(adv2) # For DNN
                    test = model_input.predict(adv2_scaled)
                    test = np.argmax(test,1) # For DNN
                    #test = model_input.predict(adv2) # For other model than DNN
                    if (test == 0): # benign break
                        adv_ex.append(adv) # adv_ex contains all adversarial examples that fool the classifier
                        breaked = True
                        break    
            index_of_mask = 0
        nb_of_needed_step = 0  
        total_ex.append(adv) # Total adversarial examples. append the final created adv ex that fool or not 
        bar()