from keras.models import model_from_json

def loaded_model():
  json_file = open('model.json','r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights('model.h5')
  
  return loaded_model
