import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

dfs = pd.read_csv("Churn_Modelling.csv")

#drop unnecessary columns
dfs.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True)

le = LabelEncoder()
ohe = OneHotEncoder()
dfs["Gender"]=le.fit_transform(dfs["Gender"])
ge_ohe = ohe.fit_transform(dfs[["Geography"]])
ge_dfs = pd.DataFrame(ge_ohe.toarray(),columns=ohe.get_feature_names_out(['Geography']))
dfs = pd.concat([dfs,ge_dfs],axis=1)

#drop geography column
dfs.drop(columns=['Geography'],inplace=True)


##Save the encoders 
with open('label-encoder-gender.pkl','wb') as file:
    pickle.dump(le,file)

with open('onehot-encoder-geography.pkl','wb') as file:
    pickle.dump(ohe,file)

X=dfs.drop(columns=['Exited'])
y=dfs['Exited']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

#Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#save the scaler
with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)


#Build ANN Model
model = Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)), #HL1
    Dense(32,activation='relu'), #HL2
    Dense(1,activation='sigmoid'), #o/p layer
])

#Initialize optimizers and loss function
opt=tf.keras.optimizers.Adam(learning_rate=0.01)
loss=tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer=opt,loss=loss,metrics=['accuracy'])

#Setup the tenser board
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

#Setup early stopping
early_stopping_callback=EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)

#Training the model
history=model.fit(
    X_train,y_train,validation_data=(X_test,y_test),epochs=100,
    callbacks=[tensorflow_callback,early_stopping_callback]
)
model.save('model.keras')

#Load tensor board extension
#tensorboard --logdir=./logs/fit/ -->run in cmd