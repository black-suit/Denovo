import numpy as np


def prepare_data(smiles,all_smile):
  all_smile_index = []
  for i in range(len(all_smile)):
    smile_index = []
    for j in range(len(all_smile[i])):
      smile_index.append(smiles.index(all_smile[i][j]))
    all_smile_index.append(smile_index)
  X_train = all_smile_index
  y_train = []
  for i in range(len(X_train)):
    x1 = X_train[i]
    x2 = x1[1:len(x1)]
    x2.append(0)
    y_train.append(x2)
    
  return X_train, y_train

def save_model(model):
  model_json = model.to_json()
  with open("model.json","w") as json_file:
    json_file.write(model_json)
  model.save_weights("model.h5")
  print("Saved model to disk")
  
if __name__ == "__main__":
  smile = zinc_data_with_bracket_original()
  valcabulary,all_smile = zinc_processed_with_bracket(smile)
  print(valcabulary)
  print(len(all_smile))
  X_train,y_train = prepare_data(valcabulary,all_smile)
  
  max_len = 81
  
  X = sequence.pad_sequences(X_train, maxlen=81, dtype='int32',
                             padding='post',truncating='pre',value=0.)
  y = sequence.pad_sequences(y_train, maxlen=81, dtype='int32',
                             padding='post',truncating='pre',value=0.)
  
  y_train_one_hot = np.array([to_categorical(sent_label, num_classes=len(valcabulary)) for sent_label in y])
  print(y_train_one_hot.shape)
  
  vocab_size = len(valcabulary)
  embed_size = len(valcabulary)
  
  N = X.shape[1]
  
  model = Sequential()
  
  model.add(Embedding(input_dim=vocab_size,,output_dim=len(valcabulary),input_length=N,mask_zero=False))
  model.add(GRU(256,input_shape=(81,64),activation='tanh',return_sequences=True))
  model.add(Dropout(0.2))
  model.add(GRU(256,activation='tanh',return_sequences=True))
  model.add(Dropout(0.2))
  model.add(TimeDistributed(Dense(embed_size,activation='softmax')))
  optimizer = Adam(lr=0.01)
  print(model.summary())
  model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
  model.fit(X,y_train_one_hot,epochs=100,batch_size=512,validation_split=0.1)
  save_model(model)
