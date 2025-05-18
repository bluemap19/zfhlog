replace_dict = {
    'CAL':['CAL', 'CALC', 'CALX', 'CALY'],
    'SP':['SP', 'Sp'],
    'GR':['GR', 'GRC'],
    'CNL':['CNL', 'CN'],
    'DEN':['DEN', 'DENC', 'RHOB'],
    'DT':['DT', 'DT24', 'DTC', 'AC', 'Ac'],
    'RXO':['RXO', 'Rxo', 'RXo', 'RS', 'Rs'],
    'RD':['RD', 'Rd', 'Rt', 'RT'],
}


def input_cols_mapping(input_cols=[], target_cols=[]):
    input_mapping_cols = input_cols.copy()

    # 遍历 l1，保留不在 l2 中的元素
    unindex_result = [element for element in input_cols if element not in target_cols]

    processed_cols = []
    for element in reversed(unindex_result):
        idx = input_mapping_cols.index(element)
        for target, replacement in replace_dict.items():
            if element in replacement:
                # 转换为集合求交集
                intersection = list(set(replacement) & set(target_cols))
                print(element, '--->', intersection)
                input_mapping_cols[idx] = intersection[0]
                processed_cols.append(element)
                break

    unprocessed_cols = [element for element in unindex_result if element not in processed_cols]
    if unprocessed_cols:
        print('Exist unprocessable cols:', unprocessed_cols)
        exit(0)

    return input_mapping_cols

# input_cols = ['DT', 'CN', 'GRC', 'DENCC', 'NMR']
input_cols = ['RTTT', 'NMR']
target_cols = ['AC', 'DEN', 'SP', 'GR', 'RT', 'CAL', 'CNL', ]
print(input_cols_mapping(input_cols=input_cols, target_cols=target_cols))