import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

from tqdm import tqdm

from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate, BatchNormalization, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model
from keras.losses import mean_squared_error as mse_loss

from keras import optimizers
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

for dirname, _, filenames in os.walk('./kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def average_imputation(df, column_name):
    # 按时间分组并求平均
    imputation = df.groupby(['timestamp'])[column_name].mean()
    df.loc[df[column_name].isnull(), column_name] = df[df[column_name].isnull()][[column_name]].apply(
        lambda x: imputation[df['timestamp'][x.index]].values)
    del imputation
    return df


'''
---通过改变数据类型减少数据所占空间
遍历每列
确定列是否为数字
确定列是否可以用整数表示
找出最小值和最大值
确定并应用适合值范围的最小数据类型
'''


def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings
            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", df[col].dtype)
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            print("min for this col: ", mn)
            print("max for this col: ", mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                df[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
                # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                        # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", df[col].dtype)
            print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return df, NAlist


def model(dense_dim_1=64, dense_dim_2=32, dense_dim_3=32, dense_dim_4=16,
          dropout1=0.2, dropout2=0.1, dropout3=0.1, dropout4=0.1, lr=0.001):
    # Inputs
    site_id = Input(shape=[1], name="site_id")
    building_id = Input(shape=[1], name="building_id")
    meter = Input(shape=[1], name="meter")
    primary_use = Input(shape=[1], name="primary_use")
    square_feet = Input(shape=[1], name="square_feet")
    year_built = Input(shape=[1], name="year_built")
    air_temperature = Input(shape=[1], name="air_temperature")
    cloud_coverage = Input(shape=[1], name="cloud_coverage")
    dew_temperature = Input(shape=[1], name="dew_temperature")
    hour = Input(shape=[1], name="hour")
    precip = Input(shape=[1], name="precip_depth_1_hr")
    weekday = Input(shape=[1], name="weekday")
    beaufort_scale = Input(shape=[1], name="beaufort_scale")
    # Embeddings layers  每一列最大的数字小于Embedding的第一个参数
    # 嵌入层将正整数（下标）转换为具有固定大小的向量
    emb_site_id = Embedding(16, 2)(site_id)
    emb_building_id = Embedding(1449, 6)(building_id)
    emb_meter = Embedding(4, 2)(meter)
    emb_primary_use = Embedding(16, 2)(primary_use)
    emb_hour = Embedding(24, 3)(hour)
    emb_weekday = Embedding(7, 2)(weekday)
    concat_emb = concatenate([
        Flatten()(emb_site_id)
        , Flatten()(emb_building_id)
        , Flatten()(emb_meter)
        , Flatten()(emb_primary_use)
        , Flatten()(emb_hour)
        , Flatten()(emb_weekday)
    ])
    categ = Dropout(dropout1)(Dense(dense_dim_1, activation='relu')(concat_emb))
    categ = BatchNormalization()(categ)
    categ = Dropout(dropout2)(Dense(dense_dim_2, activation='relu')(categ))
    # main layer
    main_l = concatenate([
        categ
        , square_feet
        , year_built
        , air_temperature
        , cloud_coverage
        , dew_temperature
        , precip
        , beaufort_scale
    ])
    main_l = Dropout(dropout3)(Dense(dense_dim_3, activation='relu')(main_l))
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(dropout4)(Dense(dense_dim_4, activation='relu')(main_l))
    # output
    output = Dense(1)(main_l)
    model = Model([site_id,
                   building_id,
                   meter,
                   primary_use,
                   square_feet,
                   year_built,
                   air_temperature,
                   cloud_coverage,
                   dew_temperature,
                   hour,
                   weekday,
                   precip,
                   beaufort_scale], output)
    model.compile(optimizer=Adam(lr=lr),
                  loss=mse_loss,
                  metrics=[root_mean_squared_error])
    return model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def get_keras_data(df, num_cols, cat_cols):
    cols = num_cols + cat_cols
    X = {col: np.array(df[col]) for col in cols}
    return X


def train_model(keras_model, X_t, y_train, batch_size, epochs, X_v, y_valid, fold, patience=3):
    early_stopping = EarlyStopping(patience=patience, verbose=1)
    model_checkpoint = ModelCheckpoint("model_" + str(fold) + ".hdf5",
                                       save_best_only=True, verbose=1, monitor='val_root_mean_squared_error',
                                       mode='min')
    hist = keras_model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs,
                           validation_data=(X_v, y_valid), verbose=1,
                           callbacks=[early_stopping, model_checkpoint])
    keras_model = load_model("model_" + str(fold) + ".hdf5",
                             custom_objects={'root_mean_squared_error': root_mean_squared_error})
    return keras_model


def main():
    # 1、处理train数据
    building_df = pd.read_csv("./kaggle/input/building_metadata.csv")
    weather_train = pd.read_csv("./kaggle/input/weather_train.csv")
    train = pd.read_csv("./kaggle/input/train.csv")
    train = train.merge(building_df, left_on="building_id", right_on="building_id", how="left")
    train = train.merge(weather_train, left_on=["site_id", "timestamp"], right_on=["site_id", "timestamp"])
    del weather_train
    # 添加一列 drop，满足条件的值为True(此时行数为20125605)
    train.loc[
        (train['meter'] == 0) & (train['site_id'] == 0) & (train['timestamp'] < '2016-05-21 00:00:00'), 'drop'] = True
    # 去掉meter为0，site_id为0且时间在20160521之前的数据(此时数据量为19777045)
    train = train[train['drop'] != True]
    # 处理时间数据(添加两列 hour,weekday)
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    train["hour"] = train["timestamp"].dt.hour
    train["weekday"] = train["timestamp"].dt.weekday  # 19777045行19列
    # 计算蒲福风级（一共从0到12级）
    train = average_imputation(train, 'wind_speed')
    beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9),
                (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]
    # item[0]就是风级数
    for item in beaufort:
        train.loc[(train['wind_speed'] >= item[1]) & (train['wind_speed'] < item[2]), 'beaufort_scale'] = item[0]
    # 去掉timestamp列，多的一列是beaufort_scale蒲福风级
    del train["timestamp"]
    le = LabelEncoder()
    # 将"primary_use"列的字符串转为数字标签 0,1,2,3....
    train["primary_use"] = le.fit_transform(train["primary_use"])
    categoricals = ["site_id", "building_id", "primary_use", "hour", "weekday", "meter"]
    drop_cols = ["sea_level_pressure", "wind_speed", "wind_direction"]
    numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",
                  "dew_temperature", "precip_depth_1_hr", "floor_count", 'beaufort_scale']
    feat_cols = categoricals + numericals
    # 两个函数会成对使用 log1p(x) := log(1+x)   expm1(x) := exp(x) - 1
    target = np.log1p(train["meter_reading"])
    del train["meter_reading"]
    # 删除列
    train = train.drop(drop_cols, axis=1)
    # (train)#15列(building_id,meter,site_id,primary_use,square_feet,year_built,floor_count,air_temperature,cloud_coverage,dew_temperature,precip_depth_1_hr,hour,weekday,beaufort_scale)

    # 2、减少训练数据所占存储空间
    train, NAlist = reduce_mem_usage(train)

    # 3、K折交叉切分训练
    oof = np.zeros(len(train))
    batch_size = 1024
    epochs = 10
    models = []
    folds = 4
    seed = 666
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold_n, (train_index, valid_index) in enumerate(kf.split(train, train['building_id'])):
        print('Fold:', fold_n)
        X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        X_t = get_keras_data(X_train, numericals, categoricals)
        X_v = get_keras_data(X_valid, numericals, categoricals)
        keras_model = model(dense_dim_1=64, dense_dim_2=32, dense_dim_3=32, dense_dim_4=16,
                            dropout1=0.2, dropout2=0.1, dropout3=0.1, dropout4=0.1, lr=0.001)
        mod = train_model(keras_model, X_t, y_train, batch_size, epochs, X_v, y_valid, fold_n, patience=3)
        models.append(mod)
        print('*' * 50)
    # 垃圾回收
    del train, target, X_train, X_valid, y_train, y_valid, X_t, X_v, kf
    gc.collect()

    # 5、处理测试集数据
    test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")
    test = test.merge(building_df, left_on="building_id", right_on="building_id", how="left")
    del building_df
    gc.collect()
    # 将primary_use这一列转换为整数
    test["primary_use"] = le.transform(test["primary_use"])
    weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")
    test = test.merge(weather_test, left_on=["site_id", "timestamp"], right_on=["site_id", "timestamp"], how="left")
    del weather_test
    # 转换时间
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    test["hour"] = test["timestamp"].dt.hour
    test["weekday"] = test["timestamp"].dt.weekday
    test = average_imputation(test, 'wind_speed')
    # 计算蒲福风级（一共从0到12级）
    for item in beaufort:
        test.loc[(test['wind_speed'] >= item[1]) & (test['wind_speed'] < item[2]), 'beaufort_scale'] = item[0]
    test = test[feat_cols]
    test, NAlist = reduce_mem_usage(test)

    # 6、测试
    i = 0
    res = []
    step_size = 50000
    for j in tqdm(range(int(np.ceil(test.shape[0] / 50000)))):
        for_prediction = get_keras_data(test.iloc[i:i + step_size], numericals, categoricals)
        res.append(np.expm1(sum([model.predict(for_prediction) for model in models]) / folds))
        i += step_size
    res = np.concatenate(res)
    submission = pd.read_csv('./kaggle/input/sample_submission.csv')
    submission['meter_reading'] = res
    submission.loc[submission['meter_reading'] < 0, 'meter_reading'] = 0
    submission.to_csv('submission.csv', index=False)
    submission


if __name__ == '__main__':
    main()
