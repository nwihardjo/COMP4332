from math import sqrt
from keras.layers import Concatenate, Dense, Dot, Dropout, Embedding, Input, Reshape
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# Function to calculate RMSE
def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))


def build_cfmodel(n_users, n_items, embed_size, output_layer='dot'):
    # YOUR CODE HERE
    
    if output_layer == 'dot':
        pass
        # YOUR CODE HERE
    elif output_layer == 'mlp':
        pass
        # YOUR CODE HERE
    else:
        raise NotImplementedError

    model = Model(inputs=[user_input, item_input],
                  outputs=model_output)
    return model


if __name__ == "__main__":
    tr_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/valid.csv")
    
    # Build User/Item vocabulary
    user_set = set(tr_df.user_id.unique())
    business_set = set(tr_df.business_id.unique())
    user_vocab = dict(zip(user_set, range(1, len(user_set) + 1)))
    user_vocab['unk'] = 0
    n_users = len(user_vocab)
    business_vocab = dict(zip(business_set, range(1, len(business_set) + 1)))
    business_vocab['unk'] = 0
    n_items = len(business_vocab)

    tr_users = tr_df.user_id.apply(lambda x: user_vocab[x] if x in user_vocab else 0).values
    tr_items = tr_df.business_id.apply(lambda x: business_vocab[x] if x in business_vocab else 0).values
    tr_ratings = tr_df.stars.values
    val_users = val_df.user_id.apply(lambda x: user_vocab[x] if x in user_vocab else 0).values
    val_items = val_df.business_id.apply(lambda x: business_vocab[x] if x in business_vocab else 0).values
    val_ratings = val_df.stars.values

    model = build_cfmodel(
        n_users, n_items, 
        embed_size=50,
        output_layer='dot')

    model.compile(optimizer='adam', loss='mse')
    
    history = model.fit(
        [tr_users, tr_items], 
        tr_ratings, 
        epochs=1, 
        verbose=1,
        callbacks=[ModelCheckpoint('model.h5')])
    y_pred = model.predict([tr_users, tr_items])
    print("TRAIN RMSE: ", rmse(y_pred, tr_ratings))
    y_pred = model.predict([val_users, val_items])
    print("VALID RMSE: ", rmse(y_pred, val_ratings))
