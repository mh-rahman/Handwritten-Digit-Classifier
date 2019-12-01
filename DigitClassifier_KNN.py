import heapq
import operator 
import numpy as np
from mnist import MNIST

def distance(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

def knn(train_images, test_images, train_labels, ind=0,k=60):
    s=test_images[ind]
    dist=[]
    for i in range(k):
        d=distance(s,train_images[i])
        dist.append(d)

    heap_arr=[(-x,y) for x,y in zip(dist,train_labels[0:k])]
    heapq.heapify(heap_arr)
    for i in range(k,len(train_images)):
            d=distance(s,train_images[i])
            if -d>heap_arr[0][0]:
                heapq.heappop(heap_arr)
                heapq.heappush(heap_arr,(-d,train_labels[i]))

    l1=[i for i in range(10)]
    l2=[0]*10
    label_count={x:y for x,y in zip(l1,l2)}
    while heap_arr:
        n_l=heapq.heappop(heap_arr)[1]
        label_count[n_l]+=1

    sorted_op = sorted(label_count.items(), key=operator.itemgetter(1))
    sorted_op.reverse()
    return sorted_op[0][0]

mndata = MNIST('Data')
train_images_data, train_labels_data = mndata.load_training()
test_images_data, test_labels_data = mndata.load_testing()
print('Data imported!')

train_images=np.asarray(train_images_data)
test_images=np.asarray(test_images_data)
train_labels=np.asarray(train_labels_data)
test_labels=np.asarray(test_labels_data)


predictions=[]
for t in range(len(test_images)):
  if t%100==0:
    print('Predictions completed: ', t)
  predictions.append(knn(train_images, test_images, train_labels, t,1))
predictions=np.asarray(predictions)

accu=[1 if predictions[i]==test_labels[i] else 0 for i in range(100)]
print(accu)

accuracy=sum(accu)/len(accu)*100
print('Accuracy = ', accuracy)