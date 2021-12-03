import re
import pandas as pd

def get_sorted_names(names:list, reverse=True):
    return sorted(names, key=len, reverse=reverse)

def get_gt_names(names:list, length:int=2):
    return [name for name in names if len(name)>=length]

def clean_shrt_eng_names(names:list, length:int=2):
    import re
    cleaned_names = [name for name in names if not re.match('[a-zA-Z]+',name) or len(name)>length]
    return cleaned_names

def get_preprocessed_names(names:list, length:int=2, reverse=True):
    gt_names = get_gt_names(names, length)
    st_names = get_sorted_names(gt_names, reverse)
    cleng_names = clean_shrt_eng_names(st_names, length)
    result = cleng_names
    return result

def drop_duplicated_names(caption:str, repName:str='홍길동', rep_count:int=2):
    return re.sub(' '.join([repName for _ in range(rep_count)], repName))
    
def replace_name(caption:str, names:list, repName:str='홍길동'):
    '''Replace name in caption to input name'''
    # Connect the names
    patterns = '|'.join(names)
    # Get cleared caption
    caption = re.sub(patterns, repName, caption)
    #caption = drop_duplicated_names(caption)
    return caption

def replace_names(data:pd.DataFrame, names:pd.DataFrame, col_data:str='caption', col_name:str='name', repName:str='홍길동', inplace=False):
    '''
    Replace names in caption dataframe from names dataframe
    data : dataframe which contains captions
    names : dataframe which contains names
    col_data : column name which contains captions
    col_name : column name whcih contains names
    repName : name for substitution
    inplace : inplace?
    '''
    _data = data if inplace else data.copy() # inplace or not
    _names = get_preprocessed_names(names[col_name])
    _data[col_data] = _data[col_data].apply(lambda x: replace_name(x, _names, repName))
    return _data