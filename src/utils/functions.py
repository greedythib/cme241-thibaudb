


def zip_dict_of_tuple(d) : 
    d1 = {k:v1 for k,(v1,_) in d.items()}
    d2 = {k:v2 for k,(_,v2) in d.items()}
    return d1,d2