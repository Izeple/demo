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