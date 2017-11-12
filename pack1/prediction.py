mypath = "/home/user/Documents/startapp/"
filename = "appDescriptions.xlsx"
trainpagename = "Examples"
classifypagename = "Classify"
from sklearn.externals import joblib
import pandas as pd

# preparing the data
from start_app_exercise import getTokenizeCleanData, prepareTrainData, filterDataWithNoEngDesc, filterDatawithEngDesc

toclassify = getTokenizeCleanData(mypath, filename, classifypagename)
filtToClassify = filterDataWithNoEngDesc(toclassify)
filtToClassify.data = filtToClassify['desc_tokens']
print(len(filtToClassify))

clf = joblib.load(mypath + 'pipeLineClassifier' + '.pkl')
print(clf)

y = clf.predict(filtToClassify.data)
filtToClassify['target'] = y

unpredictable = filterDatawithEngDesc(toclassify)
unpredictable['target'] = "Not Supported Language"

filtToClassify.drop('package', axis=1, inplace=True)
filtToClassify.drop('appName', axis=1, inplace=True)
filtToClassify.drop('description', axis=1, inplace=True)
filtToClassify.drop('desc_tokens', axis=1, inplace=True)

unpredictable.drop('package', axis=1, inplace=True)
unpredictable.drop('appName', axis=1, inplace=True)
unpredictable.drop('description', axis=1, inplace=True)
unpredictable.drop('desc_tokens', axis=1, inplace=True)

frames = [filtToClassify, unpredictable]
result = pd.concat(frames)

# print(filtToClassify.head(10))
print (unpredictable.head(3))
print()
print (filtToClassify.head(3))
# write results
result.to_csv(mypath +'output.csv', encoding='utf-8',index = False)
