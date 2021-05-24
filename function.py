#data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
    while 1:
        for key, description_list in descriptions.items():                    # วนloop key กับ des ใน  descriptions
            photo = photos[key][0]                                            # แล้วนำ key ที่ได้ ไปดึง feature ของรูปนั้นๆ มา
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, photo)  #จัดเตรียม data ให้พร้อมเข้าโมเดล ด้วยการสร้าง seq
            yield ([input_image, input_sequence], output_word)                # การส่งค่าแบบ yield ก็จะสามารถส่งค่าออกได้ โดยหลุดหรือจบการทำงานของ for loop

def data_generator_batch(descriptions, features, tokenizer, max_length, num_photos_per_batch):
    n=0
    while 1:
        for key, description_list in descriptions.items():
            n+=1
            #retrieve photo features
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature)
            if n==num_photos_per_batch:
                yield ([input_image, input_sequence], output_word)    
                n=0
                
def data_generator_list(descriptions, features, tokenizer, max_length):
    input_image = []
    input_sequence = []
    output_word = []
    for key, description_list in descriptions.items():
        #retrieve photo features
        feature = features[key][0]
        ii_, is_, ow_ = create_sequences(tokenizer, max_length, description_list, feature)
        input_image.append(ii_)
        input_sequence.append(is_)
        output_word.append(ow_)
    return input_image, input_sequence, output_word

def create_sequences(tokenizer, max_length, desc_list, photo):
    X1, X2, y = list(), list(), list()                                        # X1 คือ images, X2 คือ partial_caps และ y คือ next_words
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]                         # map คำ ไปเป็น ตัวเลข ด้วย tokenizer ที่สร้างไว้
        for i in range(1, len(seq)):                                          # for loop เพื่อสร้าง multiple X,y pairs
            in_seq, out_seq = seq[:i], seq[i]                                 # split sequencesประโยค ออกเป็น input และ output 
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]            # pad input sequence ให้มีขนาดเท่ากัน เท่ากับ max_length (pad ด้วยการเติมเลข 0)
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]    # สร้าง output sequence ด้วยการ encode คำศัพท์
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def load_all_h5_in_folder(dir, customObjects=None):
  dir_model = glob.glob(r''+dir+'*.h5')
  model_name = []
  model_list = []

  for dir in dir_model:
      temp = dir.split('/')[-1]
      model_name.append(temp)
      try:
          model = load_model(dir, custom_objects = customObjects)
          model_list.append(model)
      except:
          print("error")
  return (model_list,model_name)

# load all model
def load_h5_each_folder(dir, customObjects=None):
    model_name = []
    model_list = []
    dir_model = glob.glob(dir)
    dir_model.sort()
    dir_model = [name.split('/')[-1] for name in dir_model]

    for dir in dir_model:
        temp = os.listdir('./model/'+dir)[-1]
        temp = temp.split(".")[0]
        model_name.append(temp)
        try:
            model = load_model('./model/'+dir+'/'+ temp +'.h5',custom_objects=customObjects)
            model_list.append(model)
        except:
            print("error")

    return (model_list,model_name)