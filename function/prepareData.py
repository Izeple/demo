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