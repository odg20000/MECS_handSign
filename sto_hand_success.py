import numpy
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

data_dir = 'alldata_v10.csv'

hand = pd.read_csv(data_dir)
print(hand.head())

hand_input = hand[
    ['WRIST_X', 'THUMB_CMC_X', 'THUMB_MCP_X', 'THUMB_IP_X', 'THUMB_TIP_X', 'INDEX_FINGER_MCP_X', 'INDEX_FINGER_PIP_X',
     'INDEX_FINGER_DIP_X', 'INDEX_FINGER_TIP_X', 'MIDDLE_FINGER_MCP_X',
     'MIDDLE_FINGER_PIP_X', 'MIDDLE_FINGER_DIP_X', 'MIDDLE_FINGER_TIP_X', 'RING_FINGER_MCP_X', 'RING_FINGER_PIP_X',
     'RING_FINGER_DIP_X', 'RING_FINGER_TIP_X', 'PINKY_MCP_X', 'PINKY_PIP_X',
     'PINKY_DIP_X', 'PINKY_TIP_X', 'WRIST_Y', 'THUMB_CMC_Y', 'THUMB_MCP_Y', 'THUMB_IP_Y', 'THUMB_TIP_Y',
     'INDEX_FINGER_MCP_Y', 'INDEX_FINGER_PIP_Y', 'INDEX_FINGER_DIP_Y', 'INDEX_FINGER_TIP_Y',
     'MIDDLE_FINGER_MCP_Y', 'MIDDLE_FINGER_PIP_Y', 'MIDDLE_FINGER_DIP_Y', 'MIDDLE_FINGER_TIP_Y', 'RING_FINGER_MCP_Y',
     'RING_FINGER_PIP_Y', 'RING_FINGER_DIP_Y', 'RING_FINGER_TIP_Y', 'PINKY_MCP_Y',
     'PINKY_PIP_Y', 'PINKY_DIP_Y', 'PINKY_TIP_Y', 'WRIST_Z', 'THUMB_CMC_Z', 'THUMB_MCP_Z', 'THUMB_IP_Z', 'THUMB_TIP_Z',
     'INDEX_FINGER_MCP_Z', 'INDEX_FINGER_PIP_Z', 'INDEX_FINGER_DIP_Z',
     'INDEX_FINGER_TIP_Z', 'MIDDLE_FINGER_MCP_Z', 'MIDDLE_FINGER_PIP_Z', 'MIDDLE_FINGER_DIP_Z', 'MIDDLE_FINGER_TIP_Z',
     'RING_FINGER_MCP_Z', 'RING_FINGER_PIP_Z', 'RING_FINGER_DIP_Z',
     'RING_FINGER_TIP_Z', 'PINKY_MCP_Z', 'PINKY_PIP_Z', 'PINKY_DIP_Z', 'PINKY_TIP_Z']].to_numpy()
hand_target = hand['SIGN_TYPE'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(hand_input, hand_target, random_state=42)

"""
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
"""
train_scaled = train_input
test_scaled = test_input

sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

sc = SGDClassifier(loss='log', random_state=42)
train_score = []
test_score = []
classes = numpy.unique(train_target)

kms_count = 0
for _ in range(0, 50):
    print(kms_count)
    kms_count += 1
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

sc_log = SGDClassifier(loss='log', max_iter=50, tol=None, random_state=42)
sc_log.fit(train_scaled, train_target)
print(sc_log.score(train_scaled, train_target))
print(sc_log.score(test_scaled, test_target))

sc_hinge = SGDClassifier(loss='hinge', max_iter=50, tol=None, random_state=42)
sc_hinge.fit(train_scaled, train_target)
print(sc_hinge.score(train_scaled, train_target))
print(sc_hinge.score(test_scaled, test_target))

index = 5

print(test_scaled[index])
print(type(test_scaled[index]))
print(test_target[index])
print('sc', sc.predict(test_scaled[index].reshape(1, -1)))
print('sc_log', sc_log.predict(test_scaled[index].reshape(1, -1)))
print('sc_hinge', sc_hinge.predict(test_scaled[index].reshape(1, -1)))

joblib.dump(sc, 'sc_v10.pkl')
joblib.dump(sc_log, 'sc_log_v10.pkl')
joblib.dump(sc_hinge, 'sc_hinge_v10.pkl')

plt.plot(train_score)
plt.plot(test_score)
plt.show()

