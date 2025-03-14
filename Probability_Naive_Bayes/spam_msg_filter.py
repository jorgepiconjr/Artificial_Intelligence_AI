from collections import defaultdict
import random

# Getting Data message from file SMSSpamCollection.txt
data = [line.split("\t") for line in open("./SMSSpamCollection.txt")]
random.shuffle(data)  # Daten mischen.
upto = int(len(data)*0.8)
training_data, test_data = data[:upto], data[upto:]

labels = [label for label, msg in training_data]
n_ham = labels.count("ham")
n_spam = labels.count("spam")
p_ham = float(n_ham)/len(labels)
p_spam = float(n_spam)/len(labels)

def msg2words(msg):
    return set(msg.split())

ham, spam = defaultdict(int), defaultdict(int)
i = 0
for label, msg in training_data:
    for word in msg2words(msg):
        if label == "ham":
            ham[word] += 1
        else:
            spam[word] += 1

n_voc = len(set(list(ham.keys()) + list(spam.keys())))
def predict(msg):
    p_msg_ham = 1.0
    p_msg_spam = 1.0
    for word in msg2words(msg):
        # Berechnung der Wahrscheinlichkeiten inklusive Laplace smoothing
        p_msg_ham *= (ham[word] + 1) / (n_ham + 2)
        p_msg_spam *= (spam[word] + 1) / (n_spam + 2)
    if p_msg_ham * p_ham >= p_msg_spam * p_spam:
        return "ham"
    return "spam"

tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
for label, msg in test_data:
    predicted = predict(msg)
    if label == "spam" and predicted == "spam":
        tp += 1
    elif label == "ham" and predicted == "spam":
        fp += 1
    elif label == "ham" and predicted == "ham":
        tn += 1
    elif label == "spam" and predicted == "ham":
        fn += 1
precision = tp/(tp+fp)
recall = tp/(tp+fn)
accuracy = (tp+tn)/(tp+fp+tn+fn)

print("There are {} ({}%) ham and {} ({}%) spam SMS.".format(n_ham, round(p_ham*100, 1), n_spam, round(p_spam*100, 1)))
print("Precision is {}, recall {}, and accuracy {}".format(round(precision, 2), round(recall, 2), round(accuracy, 2)))

### Predict 2 different messages
message1 = "Dear Mr.Picon this is a formal invitation"
message2 = "Hey you are a good friend"

p1 = predict(message1)
p2 = predict(message2)
print(message1 + " = "+ p1)
print(message2 + " = "+ p2)