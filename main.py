import pandas as pd
import numpy as np
import re
from pytils import translit
from sklearn.tree import DecisionTreeClassifier
import os

custom_catalog_file = os.path.dirname(__file__)+"\\catalog.csv"
master_catalog_dt = pd.read_csv(custom_catalog_file)

ARTICLE_RATE_RANK = 0.5
TYPE_RATE_RANK = 0.8
LEN_TOTAL_ADVSUB_MIN_THRESHOLD = 1
LEN_TOTAL_ADVSUB_THRESHOLD = 4

np.random.seed(1)

validation_columns = ['NeedUserInteraction', 'NeedUserInteractionTypes',
                      'SameTypesExact', 'SameTypesTotalSub', 'SameTypesRate', 'SameArticleTotal', 'SameArticleRate']

def clean_rule(str_value=""):

    vl = str_value
    vl = re.sub('(?<=\().+?(?=\))', '', vl)
    vl = vl.replace("(", "")
    vl = vl.replace(")", "")
    vl = vl.strip()
    vl = vl.replace(".COM", "")
    vl = vl.replace(".RU", "")
    vl = vl.replace(".INFO", "")
    vl = vl.replace(".ORG", "")
    vl = vl.replace(".NET", "")
    vl = translit.translify(vl)
    vl = vl.upper()
    return vl

def cleanup(dt=None, column_name=None):

    if dt is None or not column_name:
        raise Exception("Something wrong")

    column_name1 = column_name+"_CL"

    dt[column_name1] = dt[column_name].apply(clean_rule)

    return dt

########### classification

def classify_categories(catalog_dt=None, current_dt=None):

    art_max = 4

    if catalog_dt is None or current_dt is None:
        return None

    def reprocess_dt(dt):

        fq_dt = pd.DataFrame()

        fq_dt['TypeCategory'] = None
        fq_dt['TypeCarrier'] = None
        fq_dt['TypeService'] = None

        fq_dt['TypeCategory2'] = None
        fq_dt['TypeCarrier2'] = None
        fq_dt['TypeService2'] = None

        for i_l in range(0, art_max):
            fq_dt['Article'+str(i_l+1)] = None

        res = []

        for i, row in dt.iterrows():

            article_list_str = row["Article list4"]

            if not article_list_str:
                continue

            type_category = row["TypeCategory"]
            type_carrier = row["TypeCarrier"]
            type_service = row["TypeService"]

            if type_carrier == 'ads_Exclude':
                type_carrier = ""

            if type_category == 'ads_Exclude':
                type_category = ""

            if type_service == 'ads_Exclude':
                type_service = ""

            res_s = {}
            res_s.update({

                "TypeCategory": type_category,
                "TypeService": type_service,
                "TypeCarrier": type_carrier,

            })

            a_l_list_raw = article_list_str.split(";")
            a_l_list_cl = [clean_rule(item) for item in a_l_list_raw]
            a_l_list = sorted(a_l_list_cl, key=len)

            cntr = 1
            for i_l in range(0, art_max):
                if i_l > len(a_l_list)-1:
                    break

                if cntr > art_max:
                    break

                res_s.update({
                    'Article'+str(cntr): a_l_list[i_l]
                })

                cntr = cntr + 1

            res = res + [res_s]

        fq_dt = fq_dt.append(res, ignore_index=True)
        fq_dt = fq_dt.fillna("")

        return fq_dt

    def clsfctn(X_train, X_test, y_train):

        clf = DecisionTreeClassifier()

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        return y_pred

    def retranslate_dummies(pd_dummies_Y, y_pred):

        c_n = {}
        for p_i in range(len(pd_dummies_Y.columns)):
            c_n.update({
                p_i: pd_dummies_Y.columns[p_i],
            })
        d_df = pd.DataFrame(y_pred)

        y_pred_df = d_df.rename(columns=c_n)

        res = []
        for i, row in y_pred_df.iterrows():

            n_val = [clmn for clmn in y_pred_df.columns if row[clmn] == 1]

            if n_val == "" or len(n_val) == 0:
                res = res + [np.NaN]
            else:
                res = res + n_val

        return res

    train_dt = reprocess_dt(catalog_dt)
    predict_dt = reprocess_dt(current_dt)

    ##################################

    var_columns = ['Article1', 'Article2', 'Article3', 'Article4']

    print("> processing service")

    # we need to append one to another to get the correct dummies
    res_dt = train_dt[var_columns].append(predict_dt[var_columns]).reset_index(drop=True)
    res_dt_d = pd.get_dummies(res_dt)

    # and split them again
    len_of_predict_dt = len(predict_dt[var_columns])
    X_train, X_test = res_dt_d[:-len_of_predict_dt], res_dt_d[-len_of_predict_dt:]

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train_ts = pd.get_dummies(train_dt['TypeService'])

    y_pred_ts = clsfctn(X_train, X_test, y_train_ts)

    predict_dt['TypeService2'] = retranslate_dummies(y_train_ts, y_pred_ts)

    #########

    print("> processing category")

    y_train_tcat = pd.get_dummies(train_dt['TypeCategory'])

    y_pred_tcat = clsfctn(X_train, X_test, y_train_tcat)

    predict_dt['TypeCategory2'] = retranslate_dummies(y_train_tcat, y_pred_tcat)

    #########

    print("> processing carrier")

    y_train_tc = pd.get_dummies(train_dt['TypeCarrier'])

    y_pred_tc = clsfctn(X_train, X_test, y_train_tc)

    predict_dt['TypeCarrier2'] = retranslate_dummies(y_train_tc, y_pred_tc)

    predict_dt['TypeCategory2'] = predict_dt['TypeCategory2'].fillna("E-commerce")
    predict_dt['TypeService2'] = predict_dt['TypeService2'].fillna("")
    predict_dt['TypeCarrier2'] = predict_dt['TypeCarrier2'].fillna("Desktop")

    return predict_dt

########### validation

def check_types(catalog_dt=None, row=None):

    if catalog_dt is None or row is None:
        return None

    record_exact = catalog_dt[((catalog_dt['Article list4'] == row['Article list4']) &
                            (catalog_dt['TypeCarrier'] == row['TypeCarrier2']) &
                            (catalog_dt['TypeCategory'] == row['TypeCategory2']) &
                            (catalog_dt['TypeService'] == row['TypeService2'])
                            )]

    record_differs = catalog_dt[(catalog_dt['Article list4'] == row['Article list4']) &
                                 ((catalog_dt['TypeCarrier'] != row['TypeCarrier2']) |
                            (catalog_dt['TypeCategory'] != row['TypeCategory2']) |
                            (catalog_dt['TypeService'] != row['TypeService2'])
                            )]

    record_total = catalog_dt[(catalog_dt['Article list4'] == row['Article list4'])]

    len_exact = len(record_exact)
    len_diff = len(record_differs)
    len_total = len(record_total)

    return len_exact, len_diff + len_exact, len_total


def fill_with_validation_info(catalog_dt=None, current_dt=None):

    for i, row in current_dt.iterrows():

        len_exact, len_total_advsub, len_total = check_types(catalog_dt=catalog_dt, row=row)

        ex_rate_article = 0 if len_total == 0 else len_exact / len_total
        ex_rate_types = 0 if len_total_advsub == 0 else len_exact / len_total_advsub

        if len_exact == 0 or len_total_advsub <= LEN_TOTAL_ADVSUB_MIN_THRESHOLD or (len_total_advsub > LEN_TOTAL_ADVSUB_THRESHOLD and ex_rate_types < TYPE_RATE_RANK) or ex_rate_article <= ARTICLE_RATE_RANK:
            current_dt['NeedUserInteractionTypes'][i] = True

        current_dt['SameTypesExact'][i] = len_exact
        current_dt['SameTypesTotalSub'][i] = len_total_advsub
        current_dt['SameArticleTotal'][i] = len_total

        current_dt['SameTypesRate'][i] = ex_rate_types
        current_dt['SameArticleRate'][i] = ex_rate_article

        current_dt['NeedUserInteraction'][i] = current_dt['NeedUserInteractionTypes'][i]

    return current_dt

########### data

def get_cleaned_data(validation, master_catalog_dt):

    def get_data(validation=False, master_catalog_dt=None):

        def get_catalog(master_catalog_dt):

            catalog_dt = master_catalog_dt

            if len(catalog_dt) == 0:
                return None

            return catalog_dt

        catalog_dt = get_catalog(master_catalog_dt)

        if catalog_dt is None or len(catalog_dt) == 0:
            return pd.DataFrame(), pd.DataFrame()

        mask_val = np.random.rand(len(catalog_dt)) < 0.95

        train_dt = catalog_dt.loc[mask_val].reset_index(drop=True)

        validation_dt = catalog_dt.loc[~mask_val].reset_index(drop=True)

        return train_dt, validation_dt

    catalog_dt, current_dt = get_data(validation, master_catalog_dt)

    if len(catalog_dt) == 0 or len(current_dt) == 0:
        return None

    for val_col in validation_columns:
        current_dt[val_col] = None

    current_dt['TypeCarrier2'] = None
    current_dt['TypeCategory2'] = None
    current_dt['TypeService2'] = None

    return catalog_dt, current_dt

def find_in_the_dictionary(dictionary_dt, value):

    if len(dictionary_dt) > 0:

        ddt = dictionary_dt[dictionary_dt['Word'] == value]
        if len(ddt) > 0:

            ddt = ddt.reset_index(drop=True)
            return ddt['Value'][0]

    return value


def enrich_data(validation=False):

    catalog_dt, current_dt = get_cleaned_data(validation, master_catalog_dt)

    current_dt_clsf = classify_categories(catalog_dt, current_dt)
    if current_dt_clsf is None:
        return None

    current_dt['TypeCarrier2'] = current_dt_clsf['TypeCarrier2']
    current_dt['TypeCategory2'] = current_dt_clsf['TypeCategory2']
    current_dt['TypeService2'] = current_dt_clsf['TypeService2']

    print("> Checking ...")

    current_dt = fill_with_validation_info(catalog_dt, current_dt)

    print("> Validating ...")

    for i, row in current_dt.iterrows():

        if row['TypeCarrier'] != row['TypeCarrier2'] or \
                row['TypeCategory'] != row['TypeCategory2'] or \
                row['TypeService'] != row['TypeService2']:

            current_dt['NeedUserInteractionTypes'][i] = True


    invalid_types_dt = current_dt[(current_dt['NeedUserInteractionTypes'] == True)]
    invalid_total = current_dt[(current_dt['NeedUserInteraction'] == True)]

    invalid_types = len(invalid_types_dt)
    need_interactions = len(invalid_total)

    print("Need user actions: " + str(need_interactions))

    print("Uncertain types: " + str(invalid_types))
    print("Uncertain % types: " + str(invalid_types / len(current_dt)*100))

    main_columns = ['Article list3', 'Article list4', 'TypeCarrier', 'TypeCategory', 'TypeService']

    return current_dt[main_columns + validation_columns]

res_dt = enrich_data()
exit()







