IGNORE_LOSS = -100


def read_conll_file(path):

    tokens, curr_tokens = [], []
    labels, curr_labels = [], []
    open_sent = False
    
    with open(path,encoding='UTF-8') as f:
        for line in f:  
            match len(line.split()):
                case 2:
                    open_sent = True
                case _:
                    open_sent=False
            if open_sent:
                text = line.split()[0].strip().split()
                for i in text:
                    curr_tokens.append(i) # split entities > 1 token into separate tokens and tags here
                    curr_labels.append(line.split()[-1].strip())
            else:
                if len(curr_tokens) > 0:
                    tokens.append(curr_tokens)
                    labels.append(curr_labels)

                    curr_tokens, curr_labels = [], []
                    

    return tokens, labels

