from IPython.core.display import display, HTML

import matplotlib.pyplot as plt

import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras import Model


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def attention2color(attention_score):
    r = 255 - int(attention_score * 255)
    color = rgb_to_hex((255, r, r))
    return str(color)

def visualize_attention():
    # Make new model for output predictions and attentions
    '''
    model.get_layer('attention_vec').output:
    attention_vec (Attention)    [(None, 128), (None, 54)] <- We want (None,54) that is the word att
    '''
    model_att = Model(inputs=model.input, \
                            outputs=[model.output, model.get_layer('att').output[-1]])
    idx = np.random.randint(low = 0, high=X_eval.shape[0]) # Get a random test
    tokenized_sample = np.trim_zeros(X_eval[idx]) # Get the tokenized text
    label_probs, attentions = model_att.predict([X_eval[idx:idx+1], X_eval_cons[idx:idx+1]]) # Perform the prediction

    # Get decoded text and labels
    id2word = dict(map(reversed, tokenizer.get_vocab().items()))
    decoded_text = [id2word[word] for word in tokenized_sample] 
    
    # Get classification
    label = (label_probs>0.5).astype(int).squeeze() # Only one
    label2id = ['Pos', 'Neg']

    # Get word attentions using attenion vector
    token_attention_dic = {}
    max_score = 0.0
    min_score = 0.0
    
    attentions_text = attentions[0,-len(tokenized_sample):]
    #plt.bar(np.arange(0,len(attentions.squeeze())), attentions.squeeze())
    #plt.show();
    #print(attentions_text)
    attentions_text = (attentions_text - np.min(attentions_text)) / (np.max(attentions_text) - np.min(attentions_text))
    for token, attention_score in zip(decoded_text, attentions_text):
        #print(token, attention_score)
        token_attention_dic[token] = attention_score
        

    # Build HTML String to viualize attentions
    html_text = "<hr><p style='font-size: large'><b>Text:  </b>"
    for token, attention in token_attention_dic.items():
        html_text += "<span style='background-color:{};'>{} <span> ".format(attention2color(attention),
                                                                            token)
    #html_text += "</p><br>"
    #html_text += "<p style='font-size: large'><b>Classified as:</b> "
    #html_text += label2id[label] 
    #html_text += "</p>"
    
    # Display text enriched with attention scores 
    display(HTML(html_text))
    
    # PLOT EMOTION SCORES
    _labels = ['positive', 'negative']
    probs = np.zeros(2)
    probs[1] = label_probs
    probs[0] = 1- label_probs
    plt.figure(figsize=(5,2))
    plt.bar(np.arange(len(_labels)), probs.squeeze(), align='center', alpha=0.5, color=['black', 'red', 'green', 'blue', 'cyan', "purple"])
    plt.xticks(np.arange(len(_labels)), _labels)
    plt.ylabel('Scores')
    plt.show()