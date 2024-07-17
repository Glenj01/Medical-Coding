import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
    
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    from data_util_gensim import load_data_multilabel_pre_split_for_pred,create_vocabulary,create_vocabulary_label_for_predict,get_label_sim_matrix,get_label_sub_matrix
    from model_predict_util import preprocessing,viz_attention_scores,retrieve_icd_descs,output_to_file,display_for_qualitative_evaluation,display_for_qualitative_evaluation_per_label
import time
import pickle    
import pandas as pd
from HAN_model_dynamic import HAN    
from tflearn.data_utils import pad_sequences
from gensim.models import Word2Vec  

#The model checkpoint folder to load from - choose one from the options below
#HLAN+LE+sent split trained on MIMIC-III-50
#ckpt_dir="../checkpoints/checkpoint_HAN_50_per_label_bs32_sent_split_LE/";dataset = "mimic3-ds-50";batch_size = 32;per_label_attention=True;per_label_sent_only=False;sent_split=True #HLAN trained on MIMIC-III-50

#HAN+sent split trained on MIMIC-III-Full
ckpt_dir="../checkpoints/checkpoint_HAN_sent_split_LE/";dataset = "mimic3-ds";batch_size = 128;per_label_attention=False;per_label_sent_only=False;sent_split=False #HAN trained on MIMIC-III

#other settings and hyper-parameters
word2vec_model_path = "../embeddings/processed_full.w2v"
emb_model_path = "../embeddings/word-mimic3-ds-label.model" #using the one learned from the full label sets of mimic-iii discharge summaries
label_embedding_model_path = "../embeddings/code-emb-mimic3-tr-400.model" # for label embedding initialisation (W_projection)
label_embedding_model_path_per_label = "../embeddings/code-emb-mimic3-tr-200.model" # for label embedding initialisation (per_label context_vectors)
kb_icd9 = "../knowledge_bases/kb-icd-sub.csv"

gpu=True
learning_rate = 0.01
decay_steps = 6000
decay_rate = 1.0
sequence_length = 2500
num_sentences = 100
embed_size=100
hidden_size=100
is_training=False
lambda_sim=0.0
lambda_sub=0.0
dynamic_sem=True
dynamic_sem_l2=False
multi_label_flag=True
pred_threshold=0.5
use_random_sampling=False
miu_factor=5

# ----------------------------------------------- DATA PREPROCESSING ------------------------------------------------------------
#using gpu or not - defaults to yes
if not gpu: 
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
    
#load the label list
vocabulary_word2index_label,vocabulary_index2word_label = create_vocabulary_label_for_predict(name_scope=dataset + "-HAN") # keep a distinct name scope for each model and each dataset.
if vocabulary_word2index_label == None:
    print('_label_vocabulary.pik file unavailable')
    sys.exit()

#get the number of labels
num_classes=len(vocabulary_word2index_label)

#building the vocabulary list from the pre-trained word embeddings
vocabulary_word2index, vocabulary_index2word = create_vocabulary(word2vec_model_path,name_scope=dataset + "-HAN")
vocab_size = len(vocabulary_word2index)
    
testing_data_path = "../noteInput.txt" #"../datasets.txt"
preprocess = False

if preprocess:
    #preprocess data (sentence parsing and tokenisation)
    #this is only used for a *raw* discharge summary (one each time) and when to_input is set as true.
    clinical_note_preprocessed_str = preprocessing(raw_clinical_note_file=testing_data_path,sent_parsing=sent_split,num_of_sen=100,num_of_sen_len=25) # tokenisation, padding, lower casing, sentence splitting
    output_to_file('clinical_note_temp.txt',clinical_note_preprocessed_str) #load the preprocessed data
    testX, testY = load_data_multilabel_pre_split_for_pred(vocabulary_word2index,vocabulary_word2index_label,data_path='clinical_note_temp.txt')
else:
    #this allows processing many preprocessed documents together, each in a row of the file in the testing_data_path
    testX, testY = load_data_multilabel_pre_split_for_pred(vocabulary_word2index,vocabulary_word2index_label,data_path=testing_data_path)

#padding to the maximum sequence length
testX = pad_sequences(testX, maxlen=sequence_length, value=0.)  # padding to max length

#-------------------------------------------PREDICTION AND VISUALISATION-------------------------------------------------------
#record the start time
start_time = time.time()

#create session.
config=tf.ConfigProto()
config.gpu_options.allow_growth=False
with tf.Session(config=config) as sess:
    #Instantiate Model
    model=HAN(num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,num_sentences,vocab_size,embed_size,hidden_size,is_training,lambda_sim,lambda_sub,dynamic_sem,dynamic_sem_l2,per_label_attention,per_label_sent_only,multi_label_flag=multi_label_flag)
    saver=tf.train.Saver(max_to_keep = 1) # only keep the latest model, here is the best model
    if os.path.exists(ckpt_dir+"checkpoint"):
        print("Restoring Variables from Checkpoint")
        saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))
    else:
        print("Can't find the checkpoint.going to stop")
        sys.exit()

    #get prediction results and attention scores
    if per_label_attention: # to do for per_label_sent_only
        prediction_str = display_for_qualitative_evaluation_per_label(sess,model,testX,testY,batch_size,vocabulary_index2word,vocabulary_index2word_label,sequence_length,per_label_sent_only,num_sentences=num_sentences,threshold=pred_threshold,use_random_sampling=use_random_sampling,miu_factor=miu_factor) 
    else:
        prediction_str = display_for_qualitative_evaluation(sess,model,testX,testY,batch_size,vocabulary_index2word,vocabulary_index2word_label,sequence_length=sequence_length,num_sentences=num_sentences,threshold=pred_threshold,use_random_sampling=use_random_sampling,miu_factor=miu_factor)

#prediction_str #to display raw attention score outputs with predictions
        
#sys.exit() # just to get testing time, not visualise the predictions
#print(len(prediction_str))
#get attention score and labels for visualisation
list_doc_label_marks,list_doc_att_viz,dict_doc_pred = viz_attention_scores(prediction_str)

if len(list_doc_att_viz) == 0: # if no ICD code assigned for the document.
    print('No code predicted for this document.')    
else:    
    # Preprocess the ICD data and store it in a dictionary
    icd_data = {}
    file_path = 'ICD_SNOMED_1TO1.txt'
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into individual components based on tab delimiter
            icd_code, _, _, _, _, _, _, snomed_cid, snomed_fsn, _, _, _ = line.strip().split('\t')
            
            # Store the relevant information in a dictionary
            icd_data[icd_code] = {'snomed_cid': snomed_cid, 'snomed_fsn': snomed_fsn}
    for ind, (doc_label_mark, doc_att_viz) in enumerate(zip(list_doc_label_marks,list_doc_att_viz)):
        # retrieve and display ICD-9 codes and descriptions 
        doc_label_mark_ele_list = doc_label_mark.split('-')
        if len(doc_label_mark_ele_list)==2: # HAN model, same visualisation for all codes
            doc_label_mark_without_code = '-'.join(doc_label_mark_ele_list[:3])
            print(doc_label_mark_without_code)
            filename = 'att-%s.xlsx' % (doc_label_mark[:len(doc_label_mark)-1])     
            predictions = dict_doc_pred[doc_label_mark_without_code]
            predictions = predictions.split('labels:')[0]
            ICD_9_codes = predictions.split(' ')[1:]
            #icd_code_data = load_icd_data()
            print('Predicted code list:')
            for ICD_9_code in ICD_9_codes:
                # retrieve the short title and the long title of this ICD-9 code - also potentially convert from ICD to snomed here
                print("-----------------------------------------New Code-----------------------------------------")
                if ICD_9_code in icd_data:
                    print(f"SNOMED mapping for ICD code {ICD_9_code}: {icd_data[ICD_9_code]['snomed_cid']}, {icd_data[ICD_9_code]['snomed_fsn']}")
                else:
                    icdm_data = {}
                    file_path = 'ICD_SNOMED_1TM.txt'
                    with open(file_path, 'r') as file:
                        for line in file:
                            # Split the line into individual components based on tab delimiter
                            icd_code, _, _, _, _, _, _, snomed_cid, snomed_fsn, _, _, _ = line.strip().split('\t')
                            
                            # Check if the ICD code already exists in the dictionary
                            if icd_code in icdm_data:
                                # If it does, append the SNOMED CID and FSN to the existing list
                                icdm_data[icd_code]['snomed_cid'].append(snomed_cid)
                                icdm_data[icd_code]['snomed_fsn'].append(snomed_fsn)
                            else:
                                # If it doesn't, create a new entry in the dictionary with lists for SNOMED CID and FSN
                                icdm_data[icd_code] = {'snomed_cid': [snomed_cid], 'snomed_fsn': [snomed_fsn]}
                    #_, long_tit,code_type = retrieve_icd_descs(ICD_9_code)                                                  #gets the coresponding explanation for the ICD 9 code. 
                    result = retrieve_icd_descs(ICD_9_code) #icd_code_data)
                    if result is None:
                        # Handle the case where retrieve_icd_descs returned None
                        print(f"Unable to retrieve description for ICD code {ICD_9_code}")
                    else:
                        _, long_tit, code_type = result                       
                        if ICD_9_code in icdm_data:
                            #_, long_tit,code_type = retrieve_icd_descs(ICD_9_code)
                            print(f"No direct SNOMED mappings for ICD code {ICD_9_code} ({long_tit}), Here are some options:")
                            for snomed_cid, snomed_fsn in zip(icdm_data[ICD_9_code]['snomed_cid'], icdm_data[ICD_9_code]['snomed_fsn']):
                                print(f"SNOMED_CID: {snomed_cid}, SNOMED_FSN: {snomed_fsn}")
                        else:
                            #_, long_tit,code_type = retrieve_icd_descs(ICD_9_code)
                            print("No snomed mapping present - ICD code details are:")
                            print(code_type,'code:',ICD_9_code,'(',long_tit,')')
        else: # HLAN or HA-GRU, a different visualisation for each label
            ICD_9_code = doc_label_mark_ele_list[3] # retrieve the predicted ICD-9 code
            ICD_9_code = ICD_9_code[:len(ICD_9_code)-1] # drop the trailing colon
            short_tit, long_tit,code_type = retrieve_icd_descs(ICD_9_code) #icd_code_data) # retrieve the short title and the long title of this ICD-9 code
            doc_label_mark_without_code = '-'.join(doc_label_mark_ele_list[:3])
            print(doc_label_mark_without_code,'to predict %s code' % code_type,ICD_9_code,'(%s)' % (long_tit))
            filename = 'att-%s(%s).xlsx' % (doc_label_mark[:len(doc_label_mark)-1],short_tit) #do not include the colon in the last char
            filename = filename.replace('/','').replace('<','').replace('>','') # avoid slash / or <, > signs in the filename
        
        # export the visualisation to an Excel sheet
        filename = '..\explanations\\' + filename # put the files under the ..\explanations\ folder.
        doc_att_viz.set_properties(**{'font-size': '9pt'})\
                .to_excel(filename, engine='openpyxl')
        print('Visualisation below saved to %s.' % filename) 
        
        # reset the font for the display below
        doc_att_viz.set_properties(**{'font-size': '5pt'})
        #display(doc_att_viz)
        
        #display the prediction when the label-wise visualisations for the document end
        if ind!=len(list_doc_label_marks)-1:
            #this is not the last doc label mark
            if list_doc_label_marks[ind+1][:len(doc_label_mark_without_code)] != doc_label_mark_without_code:                
                #the next doc label mark is not the current one
                print(dict_doc_pred[doc_label_mark_without_code])
                #print('Visualisation for %s ended.\n' % doc_label_mark_without_code)
        else:
            #this is the last doc label mark
            print(dict_doc_pred[doc_label_mark_without_code])                                   
            #print('Visualisation for %s ended.\n' % doc_label_mark_without_code)

print("--- The prediction and visualisation took %s seconds ---" % (time.time() - start_time))
