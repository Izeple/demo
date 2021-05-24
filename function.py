# Prepare Photo Data
# Xception
def extract_features_xception(directory,start,stop):
    model = Xception( include_top=False, pooling='avg')
    features = {}
    list_dir = os.listdir(directory)
    for img in list_dir[start:stop]:
        filename = directory + "/" + img
        try:
            image = load_img(filename, target_size=(299, 299))
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            image = image/127.5
            image = image - 1.0
        except Exception:
            pass
        
        feature = model.predict(image)
        features[img] = feature
    return features

# Extract features from each photo in the directory
def extract_features_vgg(directory,start,stop):
    model = VGG16()       
    model.layers.pop()                                                              # ลบเลเยอร์สุดท้ายที่เป็น classification ออก 
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)             # แล้วจะกลายเป็นโมเดล encoder ที่สร้าง features ของรูปภาพ
    list_dir = os.listdir(directory)

    features = dict()                                                               # สร้าง features ให้เป็น dictionary
    for name in list_dir[start:stop]:
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))                          # โหลดรูปแล้วแปลงให้มีขนาด 224*224
        image = img_to_array(image)                                                 # แปลงจาก pixels ให้เป็น numpy array
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))  # reshape ให้พร้อมเข้าโมเดล
        image = preprocess_input(image)                                             # เป็น preprocess สำเร็จรูปของ VGG  
        feature = model.predict(image, verbose=0)                                   # ให้โมเดล สร้าง features ออกมา
        image_id = name.split('.')[0]                                               # เอาชื่อไฟล์ภาพ มาเป็น key ของ features 
        features[image_id] = feature
    return features

# Extract features from each photo in the directory
def extract_features_resnet50(directory,start,stop):
    model = ResNet50(include_top=False, pooling='avg')
    features = {}
    list_dir = os.listdir(directory)
    for img in list_dir[start:stop]:
        filename = directory + "/" + img
        try:
            image = load_img(filename, target_size=(299, 299))
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            image = image/127.5
            image = image - 1.0
        except Exception:
            pass
        
        feature = model.predict(image)
        features[img] = feature
    return features

#prepareData
def get_merged_csv(flist, **kwargs):
    return pd.concat([pd.read_csv(f, **kwargs) for f in flist], axis=0)

def convert_dictList_to_dictSentence_list(dictionary,row=1000000,split=""):
    count = 1
    new_dict = dict()
    for key,value in dictionary.items():
        if count < row:
            new_dict[key] = list()
            for v in value:
                new_dict[key].append(split.join(v[1:-1]))
        count += 1
    return new_dict

def convert_dictList_to_dictSentence(dictionary,row=1000000,split=""):
    count = 1
    new_dict = dict()
    for key,value in dictionary.items():
        if count < row:
            new_dict[key] = split.join(value[1:-1])
        count += 1
    return new_dict

def show_dict(dictionary,row=1000000):
    for key in list(dictionary.keys())[:row]:
        print(dictionary[key])

# save descriptions to file, one per line
def save_descriptions(descriptions, filename, t_replace=" "):
    lines = list()
    for key, value in descriptions.items():
        lines.append(value.replace(t_replace,""))
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def split_train_test(dictionary,row):
    temp_list = list(dictionary.items())
    random.shuffle(temp_list)
    new_train_dict = dict(temp_list[:row])
    new_test_dict = dict(temp_list[row:])
    print("Train :",len(new_train_dict))
    print("Test :",len(new_test_dict))
    return new_train_dict,new_test_dict



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

# predict คำอธิบายออกมาในรูปแบบ list ที่ถูก split แล้ว และมีรูปแบบ dict ที่เก็บไว้เป็นประโยคของคำอธิบายอีกด้วย 
# โดยสามารถเลือกได้ว่าจะใช้ generate_desc เป็น argmax_search หรือ beam_search 
def predict_all_desc_dict(model, descriptions, photos, tokenizer, max_length, generate_desc):
    actual, predicted = list(), list()
    predicted_dict = dict()
    for key, desc_list in descriptions.items():
        prediction = generate_desc(model, tokenizer, photos[key], max_length)
        actual_desc = [d.split() for d in desc_list[0]]       #editz
        actual.append(actual_desc)
        predicted.append(prediction.split())
        predicted_dict[key] = prediction          # เก็บในรูปแบบ dict ด้วย
    return actual, predicted, predicted_dict

def argmax_search(model, tokenizer, photo, max_length):
    in_text = '<start>'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]     # เปลี่ยนจาก text เป็น sequences(ชุดตัวเลข)
        sequence = pad_sequences([sequence], maxlen=max_length)   # padding ในที่นี้คือ การเติม 0 ให้ input มีขนาดเท่ากัน เท่ากับ max_len
        prob = model.predict([photo,sequence], verbose=0)         # ทำนายคำ จากรูปภาพและseqก่อนหน้า
        prob = np.argmax(prob)                                       # แปลงจากความน่าจะเป็น เป็น ตัวเลขจำนวนเต็ม
        word = word_for_id(prob, tokenizer)                       # map ตัวเลข กับ คำศัพท์
        if word is None:
            break
        in_text += word
        #in_text += ' ' + word                                     # นำคำที่ถูกทำนายมาต่อท้าย
        if word == '<end>':
            break
    return in_text      

def beam_search(model, tokenizer, photo, max_length):
    in_text = '<start>'
    start = tokenizer.texts_to_sequences([in_text])[0]
    start_word = [[start, 0.0]]                       # start_word[0][0] = index of the starting word
                                                      # start_word[0][1] = probability of the word predicted
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            sequence = pad_sequences([s[0]], maxlen=max_length)   # padding ในที่นี้คือ การเติม 0 ให้ input มีขนาดเท่ากัน เท่ากับ max_len
            preds = model.predict([photo,sequence], verbose=0)    # ทำนายคำ จากรูปภาพและseqก่อนหน้า　　　#([np.array([e]), np.array(par_caps)])
            word_preds = np.argsort(preds[0])[-beam_index:]       # ทำนายออกมา top n beam_index
            for w in word_preds:                                  # สร้าง list ขึ้นมาใหม่
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])                     # เก็บ คำศัพท์ที่ทำนายได้ พร้อมกับความน่าจะเป็น
        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])   # เรียงคำศัพท์ ตามความน่าจะเป็น
        start_word = start_word[-beam_index:]                                # เอาคำศัพท์ออกมา top n beam ออกมา
    
    start_word = start_word[-1][0]
    intermediate_caption = [word_for_id(prob, tokenizer) for prob in start_word]
    final_caption = []
    for w in intermediate_caption:
        if w != '<end>':
            final_caption.append(w)
        else:
            break
    final_caption = ' '.join(final_caption[1:])
    return final_caption

# ฟังก์ชั่นที่ไว้ map ตัวเลข กลับไปเป็น คำศัพท์
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():   
        if index == integer:
            return word
    return None

def load_all_predict(dir,file):
    predict_folder = glob.glob(dir)

    dir_predict_dict = []
    for dir in predict_folder:
        dir_predict_dict.append(glob.glob(dir+file))
    dir_predict_dict = [item for sublist in dir_predict_dict for item in sublist]

    all_predicted_dict = {}
    for dir_p in dir_predict_dict:
        predicted_dict = load(open(dir_p, "rb"))
        name = re.search(r"model\w+", dir_p).group()
        all_predicted_dict[name] = predicted_dict
        print(name,"\t",[predicted_dict[k] for k in predicted_dict.keys()][0])
    return all_predicted_dict

def predict_caption(image_name=""):
    rand = random.randint(0,100)
    if image_name == "":
        image_name = list(val_dict.keys())[rand]
    img = Image.open("./pic/"+image_name)
    img = img.resize((200, 200)) 
    print(image_name)
    display(img)

    print("Actual")
    #print(val_dict[image_name])
    print(" ".join(val_dict[image_name][0]).replace("<start> ", "").replace(" <end>", ""))
    print(" ".join(val_dict[image_name][1]).replace("<start> ", "").replace(" <end>", ""))
    print(" ".join(val_dict[image_name][2]).replace("<start> ", "").replace(" <end>", ""))
    #print(convert_list_to_sentence(actual[rand]))
    print("\nPredict keras")

    for modelName in sorted(all_predicted_dict.keys()):
        try:
          print(modelName,"\t",all_predicted_dict[modelName][image_name])  
        except Exception:
          pass