# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 



df=pd.read_csv('PCOS_data.csv')
df.head()
df.shape
df.describe()
df.columns

df = df.drop(['Sl. No','Unnamed: 42','Patient File No.','Marraige Status (Yrs)','Blood Group','Hip(inch)','Waist(inch)'],axis = 1)
df['Follicle_count']=df['Follicle No. (L)'] + df['Follicle No. (R)']
df = df.drop(['Follicle No. (L)','Follicle No. (R)','Avg. F size (L) (mm)','Cycle length(days)', 'Avg. F size (R) (mm)','Endometrium (mm)','Hb(g/dl)'],axis = 1)

df["Cycle(R/I)"].replace({2: 0, 4: 1}, inplace=True)
df["Cycle(R/I)"].replace({5:0}, inplace=True)

df.rename(columns={'PCOS (Y/N)':'PCOS',
                   'Age (yrs)':'Age',
                   'Weight (Kg)':'Weight',
                   'Height(Cm)':'Height',
                   'Pulse rate(bpm)':'PulseRate',
                   'RR (breaths/min)':'RR',
                   'Pregnant(Y/N)':'Pregnant',
                   'No. of aborptions':'Abortions',
                   'FSH(mIU/mL)':'FSH',
                   'TSH (mIU/L)':'TSH',
                   'LH(mIU/mL)':'LSH',
                   'AMH(ng/mL)':'AMH',
                   'PRL(ng/mL)':'PRL',
                   'Vit D3 (ng/mL)':'VitD3',
                   'PRG(ng/mL)':'PRG',
                   'RBS(mg/dl)':'RBS',
                   'Weight gain(Y/N)':'Weight_gain',
                   'hair growth(Y/N)':'hair_growth',
                   'Skin darkening (Y/N)':'Skin_darkening',
                   'Hair loss(Y/N)':'Hair_loss',
                   'FSH/LH':'FSH_LH_ratio',
                   'Pimples(Y/N)':'Pimples',
                   'Fast food (Y/N)':'Fast_food',
                   'Reg.Exercise(Y/N)':'Reg_exercise',
                   'BP _Systolic (mmHg)':'BP_systolic',
                   'BP _Diastolic (mmHg)':'BP_diastolic',
                   'Waist:Hip Ratio':'W_H_ratio' },  
                    inplace=True)



from sklearn.model_selection import train_test_split 
X = df.iloc[:,1:].values
y= df.iloc[:,:1].values
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.3,random_state=0,stratify=df['PCOS'])


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import metrics
from sklearn.metrics import confusion_matrix


import warnings
warnings.filterwarnings('ignore')


model_rfc=rfc(n_estimators=110)
model_rfc.fit(train_X,train_y)

predictions = model_rfc.predict(test_X)

print('The accuracy of the Random Forests is',metrics.accuracy_score(predictions,test_y)*100,'%')


pickle.dump(model_rfc, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))



