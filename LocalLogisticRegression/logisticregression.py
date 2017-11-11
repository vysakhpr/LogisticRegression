from random import randint
import string 
import time
import math
import operator
import numpy as np

months=["january","february", "march","april","may","june","july","august","september","october","november","december"]
stopwords=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
flag=0
l=0

def sigmoid(x):
	return 1.0/(1 + np.exp(-x))

def logistic_regression(x,y,z,l,eta):
	w=np.copy(z)
	pred=sigmoid(np.dot(x.T,w.T))
	e=y-pred
	g=x[:,None]*e[:,None].T
	w+=(eta*g.T)-(eta*l*w)
	return w

def predict(x,z):
	w=np.copy(z)
	pred=sigmoid(np.dot(x.T,w.T))
	return pred


def random_sample(filename):
	global flag
	global l
	i=0
	f=open(filename,'r')
	if flag==0:
		for line in f:
			if "\t" not in line:
				continue
			i=i+1
		l=i
		flag=1
	n=randint(0,l-2)
	i=0
	f=open(filename,'r')
	for line in f:
		if "\t" not in line:
			continue
		if i==n:
			parts=line.split("\t")
			labels=parts[0].replace(" ","").split(',')
			document=parts[1].split('"',1)[1].rsplit('"',1)[0].replace("\\u","").translate(None,string.punctuation).translate(None,string.digits)
			words=filter(None,document.split(" "))
			words=[word for  word in words if word.lower() not in months]
			words=[word for  word in words if word.lower() not in stopwords]	
			return words,labels
		i=i+1
	return [],[]
def data_preprocess(filename):
	word_dict={}
	label_dict={}
	label_len=0
	word_len=0
	f=open(filename,'r')
	for line in f:
		if "\t" not in line:
			continue
		parts=line.split("\t")
		labels=parts[0].replace(" ","").split(',')
		document=parts[1].split('"',1)[1].rsplit('"',1)[0].replace("\\u","").translate(None,string.punctuation).translate(None,string.digits)
		words=filter(None,document.split(" "))
		words=[word for  word in words if word.lower() not in months]
		words=[word for  word in words if word.lower() not in stopwords]
		for word in words:
			if word not in word_dict:
				word_dict[word]=word_len
				word_len=word_len+1
			# else:
			# 	word_dict[word]=word_dict[word]+1	
		for label in labels:
			if label not in label_dict:
				label_dict[label]=label_len
				label_len=label_len+1
	return word_dict,label_dict

features,label_dict=data_preprocess("/scratch/ds222-2017/assignment-1/DBPedia.small/small_training.txt")
n=len(features)
m=len(label_dict)
w=np.zeros((m,n+1))
loss=""
k=0
tmp=[]
alpha=0.01
start=time.time()
while True:
	x=np.zeros(n+1)
	x[n]=1
	y=np.zeros(m)
	[words,labels]=random_sample("/scratch/ds222-2017/assignment-1/DBPedia.small/small_training.txt")
	# for label in labels:
	# 	y[label_dict[label]]=1
	y[label_dict[labels[0]]]=1
	for word in words:
		x[features[word]]=1
	w1=logistic_regression(x,y,w,0,alpha)
	alpha=alpha/(k+1)
	k=k+1
	t=np.linalg.norm(np.subtract(w1,w))
	if(k%l==0):
		loss = loss + "Epoch #:"+str(t)+"\n"
	# tmp.append(t)
	# if len(tmp)==50:
	# 	print min(tmp)
	# 	tmp=[]
	# if t < 0.0001:
	if k/l>5:
		break
	w=np.copy(w1)
print loss
finish=time.time()
print "Finished Training in "+str(finish- start)+ "seconds"+"in"+str(k)+"iterations"
print "Training Data Size: "+str(l)

start=time.time()
s=0
t=0
f=open("/scratch/ds222-2017/assignment-1/DBPedia.small/small_test.txt",'r')
for line in f:
	if "\t" not in line:
		continue
	parts=line.split("\t")
	labels=parts[0].replace(" ","").split(',')
	document=parts[1].split('"',1)[1].rsplit('"',1)[0].replace("\\u","").translate(None,string.punctuation).translate(None,string.digits)
	words=filter(None,document.split(" "))
	words=[word for  word in words if word.lower() not in months]
	words=[word for  word in words if word.lower() not in stopwords]
	x=np.zeros(n+1)
	x[n]=1
	y=np.zeros(m)
	for word in words:
		x[features[word]]=1
	y=predict(x,w1)
	i=np.argmax(y)
	for lb in labels:
		if label_dict[lb]==i:
			t=t+1
			break
	s=s+1
finish=time.time()
print "Finished Testing in "+str(finish- start)+ "seconds"
print t*1.0/s







